#include "metal/MetalBackend.hpp"
#include "metal/abi/GgufRepack.h"
#include "vulkan/abi/Linear.h"

#include <vulkan/vulkan.h>
#include <openssl/sha.h>
#include "vulkan/native/G17GPairData.hpp"
#include "vulkan/native/G17GSingleData.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <thread>
#include <unistd.h>

namespace splash::metal {
namespace {

using Clock = std::chrono::steady_clock;

void checked(VkResult result, const char *operation) {
  if (result != VK_SUCCESS)
    throw MetalBackendError(std::string(operation) + ": Vulkan result " + std::to_string(result));
}

uint64_t aligned(uint64_t value, uint64_t alignment) {
  if (value > UINT64_MAX - alignment + 1) throw MetalBackendError("allocation size overflow");
  return (value + alignment - 1) / alignment * alignment;
}

bool supportedFormat(uint32_t f) {
  return f == GGUF_FMT_Q4K || f == GGUF_FMT_Q5K || f == GGUF_FMT_Q6K ||
         f == GGUF_FMT_Q80 || f == GGUF_FMT_IQ4XS || f == GGUF_FMT_IQ4NL ||
         quant_affine_format(f) || f == GGUF_FMT_BF16 || f == GGUF_FMT_F16;
}

static bool g17gPairRequested(){const auto*v=std::getenv("SPLASH_VK_G17G_PAIR");return v&&std::strcmp(v,"1")==0;}
static bool g17gSingleRequested(){const auto*v=std::getenv("SPLASH_VK_G17G_SINGLE");return v&&std::strcmp(v,"1")==0;}
static bool g17gRequested(){return g17gPairRequested()||g17gSingleRequested();}

struct Allocation;

struct Device {
  VkInstance instance = VK_NULL_HANDLE;
  VkPhysicalDevice physical = VK_NULL_HANDLE;
  VkDevice device = VK_NULL_HANDLE;
  VkQueue queue = VK_NULL_HANDLE;
  uint32_t queueFamily = 0;
  uint32_t timestampBits = 0;
  VkPhysicalDeviceProperties properties{};
  VkPhysicalDeviceMemoryProperties memory{};
  DeviceCapabilities capabilities;
  bool nativeBfloatMatrix16=false;
  VkDescriptorSetLayout descriptorLayout = VK_NULL_HANDLE;
  VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
  VkPipelineCache pipelineCache = VK_NULL_HANDLE;
  VkPipelineLayout nativePairLayout = VK_NULL_HANDLE;
  VkShaderEXT nativePairShaders[2] = {VK_NULL_HANDLE,VK_NULL_HANDLE};
  bool shaderObjectsEnabled = false;
  bool nativePairsReady = false;
  PFN_vkCreateShadersEXT createShaders = nullptr;
  PFN_vkDestroyShaderEXT destroyShader = nullptr;
  PFN_vkCmdBindShadersEXT bindShaders = nullptr;

  void loadNativePairs() {
    if(!shaderObjectsEnabled||shaderDirectory.empty())return;
    createShaders=reinterpret_cast<PFN_vkCreateShadersEXT>(vkGetDeviceProcAddr(device,"vkCreateShadersEXT"));
    destroyShader=reinterpret_cast<PFN_vkDestroyShaderEXT>(vkGetDeviceProcAddr(device,"vkDestroyShaderEXT"));
    bindShaders=reinterpret_cast<PFN_vkCmdBindShadersEXT>(vkGetDeviceProcAddr(device,"vkCmdBindShadersEXT"));
    if(!createShaders||!destroyShader||!bindShaders)return;
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT,0,8};
    VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    layout.pushConstantRangeCount=1;layout.pPushConstantRanges=&push;
    checked(vkCreatePipelineLayout(device,&layout,nullptr,&nativePairLayout),"create native pair layout");
    try {
      for(uint32_t fmt:{3u,4u}) {
        auto path=shaderDirectory/("native-f"+std::to_string(fmt)+".bin");
        std::ifstream file(path,std::ios::binary|std::ios::ate);
        if(!file)throw MetalBackendError("qualified native pair binary missing");
        const auto size=file.tellg();if(size<80||size>1048576)throw MetalBackendError("invalid native pair binary size");
        std::vector<unsigned char> binary(static_cast<size_t>(size));file.seekg(0);
        file.read(reinterpret_cast<char*>(binary.data()),binary.size());
        if(!file)throw MetalBackendError("truncated native pair binary");
        unsigned char digest[SHA256_DIGEST_LENGTH];SHA256(binary.data(),binary.size(),digest);
        static constexpr char digits[]="0123456789abcdef";std::string hash;
        for(auto b:digest){hash+=digits[b>>4];hash+=digits[b&15];}
        if(hash!=(fmt==3?kNativeQ5Sha:kNativeQ6Sha))throw MetalBackendError("native pair binary does not match qualified package");
        VkShaderCreateInfoEXT info{VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT};
        info.stage=VK_SHADER_STAGE_COMPUTE_BIT;info.codeType=VK_SHADER_CODE_TYPE_BINARY_EXT;
        info.codeSize=binary.size();info.pCode=binary.data();info.pName="main";
        info.pushConstantRangeCount=1;info.pPushConstantRanges=&push;
        checked(createShaders(device,1,&info,nullptr,&nativePairShaders[fmt-3]),"import native pair shader object");
      }
      nativePairsReady=true;
      std::fprintf(stderr,"G17G Q5/Q6 native shader objects: qualified package loaded\n");
    } catch(const std::exception&e) {
      for(auto &shader:nativePairShaders)if(shader){destroyShader(device,shader,nullptr);shader=VK_NULL_HANDLE;}
      std::fprintf(stderr,"G17G native pair unavailable; using ordinary Vulkan: %s\n",e.what());
    }
  }

  VkShaderEXT nativeSingleShaders[8] = {};
  bool nativeSinglesReady = false;

  void loadNativeSingles() {
    if(!g17gSingleRequested()||!shaderObjectsEnabled||!nativePairLayout)return;
    VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT,0,8};
    try {
      for(uint32_t i=0;i<8;++i){
        const auto &data=kNativeSingles[i];
        auto path=shaderDirectory/(std::string("native-single-")+data.key+".bin");
        std::ifstream file(path,std::ios::binary|std::ios::ate);
        if(!file)throw MetalBackendError("qualified native single binary missing");
        const auto size=file.tellg();if(size<80||size>1048576)throw MetalBackendError("invalid native single binary size");
        std::vector<unsigned char> binary(static_cast<size_t>(size));file.seekg(0);file.read(reinterpret_cast<char*>(binary.data()),binary.size());
        if(!file)throw MetalBackendError("truncated native single binary");
        unsigned char digest[SHA256_DIGEST_LENGTH];SHA256(binary.data(),binary.size(),digest);
        static constexpr char digits[]="0123456789abcdef";std::string hash;
        for(auto b:digest){hash+=digits[b>>4];hash+=digits[b&15];}
        if(hash!=data.sha)throw MetalBackendError("native single binary does not match qualified package");
        VkShaderCreateInfoEXT info{VK_STRUCTURE_TYPE_SHADER_CREATE_INFO_EXT};info.stage=VK_SHADER_STAGE_COMPUTE_BIT;info.codeType=VK_SHADER_CODE_TYPE_BINARY_EXT;
        info.codeSize=binary.size();info.pCode=binary.data();info.pName="main";info.pushConstantRangeCount=1;info.pPushConstantRanges=&push;
        checked(createShaders(device,1,&info,nullptr,&nativeSingleShaders[i]),"import native single shader object");
      }
      nativeSinglesReady=true;std::fprintf(stderr,"G17G Q5/Q6 native single shader objects: package loaded\n");
    }catch(const std::exception &e){
      for(auto &shader:nativeSingleShaders)if(shader){destroyShader(device,shader,nullptr);shader=VK_NULL_HANDLE;}
      std::fprintf(stderr,"G17G native singles unavailable; using ordinary Vulkan: %s\n",e.what());
    }
  }

  VkShaderEXT nativeSingle(const ComputeDispatch &dispatch) {
    if(!dispatch.pipelineName.starts_with("vulkan_gguf_single_g17g_"))return VK_NULL_HANDLE;
    if(!nativeSinglesReady)throw MetalBackendError("native single shaders unavailable");
    const auto bytes=std::find_if(dispatch.bytes.begin(),dispatch.bytes.end(),[](const auto&b){return b.index==12;});
    const auto root=std::find_if(dispatch.bytes.begin(),dispatch.bytes.end(),[](const auto&b){return b.index==22;});
    if(bytes==dispatch.bytes.end()||bytes->sizeBytes!=sizeof(VulkanLinearParams)||!bytes->data||root==dispatch.bytes.end()||root->sizeBytes!=208||!root->data)throw MetalBackendError("incomplete native single parameters");
    VulkanLinearParams p;std::memcpy(&p,bytes->data,sizeof(p));
    if(p.rows!=8||p.active_rows!=8||p.row_start||p.input_f32||p.input_size%256||!p.input_size||p.plane_input_size!=p.input_size||!p.out_stride||p.out_stride%64||p.cols[0]!=p.out_stride||p.cols[1]||p.cols[2]||p.offset[0]||(p.fmt[0]!=3&&p.fmt[0]!=4)||p.epilogue>2||(p.epilogue&&p.output_f32))throw MetalBackendError("unsupported native single geometry");
    const uint32_t i=(p.fmt[0]-3)*4+(p.epilogue?p.epilogue+1:p.output_f32?1:0);const auto&data=kNativeSingles[i];
    if(dispatch.pipelineName!=std::string("vulkan_gguf_single_g17g_")+data.key)throw MetalBackendError("native single epilogue mismatch");
    const auto buffer=[&](uint32_t slot)->const MetalBuffer&{for(const auto&b:dispatch.buffers)if(b.index==slot)return b.buffer;throw MetalBackendError("native single buffer missing");};
    if(buffer(20).sizeBytes()!=20||!buffer(20).contents())throw MetalBackendError("native single runtime parameters missing");
    uint32_t params[5];std::memcpy(params,buffer(20).contents(),20);const uint32_t splits=params[1];
    if(params[0]!=p.input_size||!splits||splits>8||(splits&(splits-1))||p.input_size%(32*splits)||params[2]!=p.out_stride||params[3]||params[4]!=1)throw MetalBackendError("native single runtime parameters mismatch");
    if(dispatch.threadsPerThreadgroup.x!=64||dispatch.threadsPerThreadgroup.y!=1||dispatch.threadsPerThreadgroup.z!=1||dispatch.threadgroups.x!=p.out_stride/64||dispatch.threadgroups.y!=splits||dispatch.threadgroups.z!=1||dispatch.indirectThreadgroups.buffer)throw MetalBackendError("native single grid mismatch");
    const uint64_t outputBytes=uint64_t(8)*p.out_stride*(p.output_f32?4:2);
    if(buffer(0).sizeBytes()<uint64_t(8)*p.input_size*2||buffer(10).sizeBytes()<outputBytes||buffer(15).sizeBytes()<uint64_t(8)*p.out_stride*splits*4||buffer(16).sizeBytes()<uint64_t(p.out_stride/64)*4||(p.epilogue&&buffer(11).sizeBytes()<uint64_t(8)*p.out_stride*2))throw MetalBackendError("native single buffer extent mismatch");
    const auto overlap=[](const MetalBuffer&a,const MetalBuffer&b){return a.gpuAddress()<b.gpuAddress()+b.sizeBytes()&&b.gpuAddress()<a.gpuAddress()+a.sizeBytes();};
    if(overlap(buffer(0),buffer(10)))throw MetalBackendError("native single activation alias");
    const uint64_t stride=uint64_t(p.input_size)/32*16,rows=buffer(1).sizeBytes()/stride;
    if(buffer(1).sizeBytes()%stride||rows%256||rows<p.out_stride||buffer(2).sizeBytes()!=rows*p.input_size/32*(p.fmt[0]==3?4:8)||buffer(3).sizeBytes()!=rows*p.input_size/256*(p.fmt[0]==3?16:20))throw MetalBackendError("native single canonical plane extent mismatch");
    for(uint32_t slot=1;slot<=3;++slot)for(uint32_t a:{0u,10u,15u,16u})if(overlap(buffer(slot),buffer(a)))throw MetalBackendError("native single weight alias");
    std::array<uint32_t,52> words;std::memcpy(words.data(),root->data,208);std::vector<uint32_t> slots={0,1,2,3,10,15,16};if(p.epilogue)slots.push_back(11);slots.push_back(20);
    for(uint32_t j=0;j<slots.size();++j)if((uint64_t(words[2*j])|(uint64_t(words[2*j+1])<<32))!=buffer(slots[j]).gpuAddress())throw MetalBackendError("native single Root pointer mismatch");
    const uint32_t scalars[13]={p.input_size&~255u,p.input_size*8,p.input_size*8+16,splits-1,0,p.input_size,splits,0,p.input_size/32,p.input_size/32/splits,1,0,p.out_stride};
    if(std::memcmp(words.data()+data.rootBase,scalars,sizeof(scalars))||std::memcmp(words.data()+data.rootBase+13,data.literals,data.literalCount*4))throw MetalBackendError("native single Root scalar mismatch");
    return nativeSingleShaders[i];
  }

  VkShaderEXT nativePair(const ComputeDispatch &dispatch) {
    if(auto single=nativeSingle(dispatch))return single;
    if(dispatch.pipelineName!="vulkan_gguf_pair_g17g_f3"&&dispatch.pipelineName!="vulkan_gguf_pair_g17g_f4")return VK_NULL_HANDLE;
    if(!nativePairsReady)throw MetalBackendError("native pair shaders are unavailable");
    const uint32_t fmt=dispatch.pipelineName.ends_with("f3")?3:4;
    const auto bytes=std::find_if(dispatch.bytes.begin(),dispatch.bytes.end(),[](const auto&v){return v.index==12;});
    const auto root=std::find_if(dispatch.bytes.begin(),dispatch.bytes.end(),[](const auto&v){return v.index==22;});
    if(bytes==dispatch.bytes.end()||!bytes->data||bytes->sizeBytes!=sizeof(VulkanLinearParams)||root==dispatch.bytes.end()||!root->data||root->sizeBytes!=208)
      throw MetalBackendError("incomplete native pair parameters");
    VulkanLinearParams p;std::memcpy(&p,bytes->data,sizeof(p));
    if(p.rows!=8||p.active_rows!=8||p.row_start||p.input_f32||p.output_f32||!p.input_size||p.input_size%256||p.plane_input_size!=p.input_size||!p.out_stride||p.out_stride%32||p.cols[0]!=p.out_stride||p.cols[1]||p.cols[2]||p.offset[0]||p.fmt[0]!=fmt)
      throw MetalBackendError("unsupported native pair geometry");
    if(dispatch.threadsPerThreadgroup.x!=32||dispatch.threadsPerThreadgroup.y!=1||dispatch.threadsPerThreadgroup.z!=1||dispatch.threadgroups.x!=p.out_stride/32||dispatch.threadgroups.y!=1||dispatch.threadgroups.z!=1||dispatch.indirectThreadgroups.buffer)
      throw MetalBackendError("native pair grid mismatch");
    const auto buffer=[&](uint32_t slot)->const MetalBuffer&{auto i=std::find_if(dispatch.buffers.begin(),dispatch.buffers.end(),[&](const auto&v){return v.index==slot;});if(i==dispatch.buffers.end())throw MetalBackendError("missing native pair buffer");return i->buffer;};
    if(buffer(0).sizeBytes()<uint64_t(8)*p.input_size*2||buffer(10).sizeBytes()<uint64_t(8)*p.out_stride*2||buffer(11).sizeBytes()<uint64_t(8)*p.out_stride*2||buffer(20).sizeBytes()!=20)
      throw MetalBackendError("native pair activation extent mismatch");
    const auto overlaps=[](const MetalBuffer&a,const MetalBuffer&b){return a.gpuAddress()<b.gpuAddress()+b.sizeBytes()&&b.gpuAddress()<a.gpuAddress()+a.sizeBytes();};
    if(overlaps(buffer(0),buffer(10))||overlaps(buffer(0),buffer(11))||overlaps(buffer(10),buffer(11)))throw MetalBackendError("native pair activation alias");
    for(uint32_t start:{1u,4u}){
      const uint64_t stride=uint64_t(p.input_size)/32*16,rows=buffer(start).sizeBytes()/stride;
      if(buffer(start).sizeBytes()%stride||rows%256||rows<p.out_stride||buffer(start+1).sizeBytes()!=rows*p.input_size/32*(fmt==3?4:8)||buffer(start+2).sizeBytes()!=rows*p.input_size/256*(fmt==3?16:20))
        throw MetalBackendError("native pair canonical plane extent mismatch");
      for(uint32_t slot=start;slot<start+3;++slot)for(uint32_t activation:{0u,10u,11u})if(overlaps(buffer(slot),buffer(activation)))throw MetalBackendError("native pair weight alias");
    }
    std::array<uint32_t,52> words;std::memcpy(words.data(),root->data,208);
    constexpr uint32_t slots[11]={0,1,2,3,4,5,6,10,11,0,20};
    for(uint32_t i=0;i<11;++i)if((uint64_t(words[2*i])|(uint64_t(words[2*i+1])<<32))!=buffer(slots[i]).gpuAddress())throw MetalBackendError("native pair Root pointer mismatch");
    if(words[34]!=p.out_stride||std::memcmp(words.data()+35,fmt==3?kNativeQ5Literals:kNativeQ6Literals,52))throw MetalBackendError("native pair Root constants mismatch");
    const uint32_t expected[5]={p.input_size,1,p.out_stride,0,1};
    if(!buffer(20).contents()||std::memcmp(buffer(20).contents(),expected,20))throw MetalBackendError("native pair runtime parameters mismatch");
    return nativePairShaders[fmt-3];
  }

  std::map<std::string, VkPipeline> pipelines;
  std::filesystem::path shaderDirectory;
  mutable std::mutex mutex;
  std::mutex queueMutex;
  std::mutex callbackMutex;
  std::vector<std::weak_ptr<Allocation>> allocations;
  MetalMemoryStats stats;
  std::atomic<bool> inFlight = false;
  std::atomic<bool> gpuDone = true;
  std::atomic<bool> stopped = false;
  std::atomic<bool> healthy = true;
  std::atomic<uint64_t> generation = 0;
  std::atomic<int64_t> startedNs = 0;
  double timeout = kCommandTimeoutSeconds;
  std::string reason;
  std::function<void()> operationGuard;
  std::function<bool()> waitInterrupt;

  ~Device() {
    if (device) {
      vkDeviceWaitIdle(device);
      for(auto shader:nativeSingleShaders)if(shader&&destroyShader)destroyShader(device,shader,nullptr);
      for(auto shader:nativePairShaders)if(shader&&destroyShader)destroyShader(device,shader,nullptr);
      if(nativePairLayout)vkDestroyPipelineLayout(device,nativePairLayout,nullptr);
      for (auto &[key, pipeline] : pipelines) vkDestroyPipeline(device, pipeline, nullptr);
      if (pipelineCache) vkDestroyPipelineCache(device, pipelineCache, nullptr);
      if (pipelineLayout) vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
      if (descriptorLayout) vkDestroyDescriptorSetLayout(device, descriptorLayout, nullptr);
      vkDestroyDevice(device, nullptr);
    }
    if (instance) vkDestroyInstance(instance, nullptr);
  }

  void fail(std::string message) {
    std::lock_guard lock(mutex);
    healthy = false;
    if (reason.empty()) reason = std::move(message);
  }

  void check() const {
    if (!healthy.load()) {
      std::lock_guard lock(mutex);
      throw MetalBackendError(reason);
    }
    if (stopped.load()) throw MetalBackendError("Vulkan backend stopped");
    if (operationGuard) operationGuard();
  }

  VkPipeline pipeline(const ComputeDispatch &dispatch) {
    const auto threads = dispatch.threadsPerThreadgroup;
    if (!threads.x || !threads.y || !threads.z ||
        threads.x > properties.limits.maxComputeWorkGroupSize[0] ||
        threads.y > properties.limits.maxComputeWorkGroupSize[1] ||
        threads.z > properties.limits.maxComputeWorkGroupSize[2] ||
        threads.x * threads.y * threads.z > properties.limits.maxComputeWorkGroupInvocations)
      throw MetalBackendError("invalid Vulkan threadgroup dimensions");
    if (dispatch.pipelineName.empty() || dispatch.pipelineName.find_first_not_of(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") != std::string::npos)
      throw MetalBackendError("invalid Vulkan pipeline name");
    std::string key = dispatch.pipelineName + ":" + std::to_string(threads.x) + ":" +
                            std::to_string(threads.y) + ":" + std::to_string(threads.z);
    std::string source = dispatch.pipelineName;
    std::vector<uint32_t> constants{uint32_t(threads.x), uint32_t(threads.y), uint32_t(threads.z)};
    if (source == "decode_accept_dflash") {
      constants.push_back(vulkanVerifyRows()); key += ":verify" + std::to_string(vulkanVerifyRows());
    }
    if (source == "draft_conv" || source == "draft_conv_h2048") {
      constants.push_back(source.ends_with("_h2048") ? 2048 : 5120); source = "draft_conv";
    }
    if (source == "prefill_draft_context_kv" || source == "draft_context_kv_commit") {
      constants.push_back(source == "draft_context_kv_commit"); source = "draft_context_kv";
    }
    if (source == "prefill_draft_context_store" || source == "prefill_draft_context_load") {
      constants.push_back(source.ends_with("_load")); source = "draft_context_window";
    }
    if (source.starts_with("prefill_attention_") || source.starts_with("verify_attention_")) {
      const bool verify = source.starts_with("verify_");
      const bool small = source.find("_kv2_g8") != std::string::npos;
      constants.push_back(verify); constants.push_back(small ? 2 : 4);
      constants.push_back(small ? 16 : 24); constants.push_back(source.ends_with("_f32"));
      if (source.find("_q8_") != std::string::npos || source.find("table") != std::string::npos)
        throw MetalBackendError("Vulkan attention requires BF16 KV and plain projection input");
      if (source.find("_qkv") != std::string::npos) source = "attention_qkv";
      else if (source.find("_gate") != std::string::npos) source = "attention_gate";
      else if (source.find("_store") != std::string::npos) source = "attention_store";
      else if (source.find("_split") != std::string::npos) {
        bool onePage=false;
        const char *matrixAttention=std::getenv("SPLASH_VK_ATTENTION_PAGE32");
        if(nativeBfloatMatrix16 && matrixAttention && std::strcmp(matrixAttention,"1")==0){
          // A local accumulator avoids spilling across QK/softmax. Every
          // split must contain at most one page; retain scalar fallback otherwise.
          for(const auto &bytes:dispatch.bytes){
            if(bytes.index!=(verify?7u:4u)||!bytes.data)continue;
            if(verify){
              const size_t lanes=dispatch.threadgroups.z;
              if(!lanes||lanes>4||bytes.sizeBytes<lanes*24)continue;
              onePage=true;
              for(size_t lane=0;lane<lanes;++lane){
                uint32_t meta[6];std::memcpy(meta,static_cast<const char*>(bytes.data)+lane*24,24);
                const uint64_t pages=(uint64_t(meta[0])+8+31)/32;
                onePage=onePage&&meta[4]&&pages<=meta[4];
              }
            }else if(bytes.sizeBytes>=28){
              uint32_t meta[7];std::memcpy(meta,bytes.data,28);
              const uint64_t pages=(uint64_t(meta[0])+meta[1]+31)/32;
              onePage=meta[1]&&meta[6]&&pages<=meta[6];
            }
          }
        }
        source=onePage?"attention_split_page32":"attention_split";
        if(onePage)key+=":page32";
      }
      else if (source.find("_reduce") != std::string::npos) source = "attention_reduce";
      else throw MetalBackendError("unknown Vulkan attention pipeline");
    }
    if (source.starts_with("prefill_gdn_") || source.starts_with("verify_gdn_fused")) {
      constants.push_back(source.find("_f32") != std::string::npos);
      constants.push_back(source.find("_vh32") != std::string::npos ? 32 : 48);
      constants.push_back(source.ends_with("_bf16_state"));
      if (source.starts_with("verify_gdn_fused")) {
        if (source.find("table") != std::string::npos) throw MetalBackendError("Vulkan GDN uses plain projection input");
        source = "verify_gdn_fused";
      } else if (source.starts_with("prefill_gdn_prepare")) source = "prefill_gdn_prepare";
      else if (source.starts_with("prefill_gdn_scan")) source = "prefill_gdn_scan";
      else source = "prefill_gdn_gate";
    }
    if (source.starts_with("gguf_embed_")) {
      bool f32 = source.starts_with("gguf_embed_f32_");
      const std::string name = source.substr(f32 ? 15 : 11);
      uint32_t format = 0;
      while (format < GGUF_FMT_COUNT && name != kQuantFormats[format].name) ++format;
      if (!supportedFormat(format)) throw MetalBackendError("embedding format is not ported: " + name);
      constants.push_back(format);
      constants.push_back(f32);
      source = "gguf_embed";
    }
    if (source.starts_with("vulkan_test_dequant_")) {
      const std::string name = source.substr(std::strlen("vulkan_test_dequant_"));
      uint32_t format = 0;
      while (format < GGUF_FMT_COUNT && name != kQuantFormats[format].name) ++format;
      if (!supportedFormat(format)) throw MetalBackendError("dequantization format is not ported");
      constants.push_back(format);
      source = "vulkan_test_dequant";
    }
    if(source=="vulkan_register_reduce"){
      const auto bytes=std::find_if(dispatch.bytes.begin(),dispatch.bytes.end(),[](const auto &b){return b.index==12;});
      if(bytes==dispatch.bytes.end()||!bytes->data||bytes->sizeBytes!=sizeof(VulkanLinearParams))throw MetalBackendError("invalid register reduction parameters");
      VulkanLinearParams p;std::memcpy(&p,bytes->data,sizeof(p));constants.push_back(vulkanRegisterSplits(p.input_size,p.out_stride));key+=":splits"+std::to_string(vulkanRegisterSplits(p.input_size,p.out_stride));
    }
    if (source == "vulkan_gguf_linear" || source == "vulkan_gguf_coop" || source == "vulkan_gguf_register" || source == "vulkan_gguf_register16" || source == "vulkan_gguf_staged16" || source == "vulkan_gguf_direct16") {
      const bool registered16=source=="vulkan_gguf_register16";
      const bool registered=source=="vulkan_gguf_register"||registered16;
      const bool staged=source=="vulkan_gguf_staged16";
      const bool direct=source=="vulkan_gguf_direct16";
      const bool native=source!="vulkan_gguf_linear";
      if(registered&&!(registered16?capabilities.nativeBfloatMatrix16:capabilities.nativeBfloatMatrix8))throw MetalBackendError("required BF16 matrix mode is unavailable");
      if(native&&!staged&&!direct&&!capabilities.nativeMatrixSupported)throw MetalBackendError("native 8x8 matrix capability is unavailable");
      if((staged||direct)&&!capabilities.nativeMixedMatrix16Direct)throw MetalBackendError("qualified BF16/FP16 16x16 matrix capability is unavailable");
      const auto bytes = std::find_if(dispatch.bytes.begin(), dispatch.bytes.end(), [](const auto &b) { return b.index == 12; });
      if (bytes == dispatch.bytes.end() || !bytes->data || bytes->sizeBytes != sizeof(VulkanLinearParams))
        throw MetalBackendError("invalid Vulkan projection parameters");
      VulkanLinearParams p;
      std::memcpy(&p, bytes->data, sizeof(p));
      if (!p.rows || !p.input_size || p.input_size % 32 || p.plane_input_size < p.input_size ||
          p.plane_input_size % 32 || !p.out_stride || p.epilogue > 2 || p.output_f32 > 1 ||
          !p.active_rows || p.active_rows > 8 || !p.cols[0] ||
          (native ? (threads.x % 32 || threads.x > 256) : (threads.x != 32 && threads.x != 256)) || threads.y != 1 || threads.z != 1)
        throw MetalBackendError("invalid Vulkan projection geometry");
      const auto buffer = [&](uint32_t index) -> const MetalBuffer & {
        auto found = std::find_if(dispatch.buffers.begin(), dispatch.buffers.end(), [index](const auto &b) { return b.index == index; });
        if (found == dispatch.buffers.end()) throw MetalBackendError("missing Vulkan projection binding");
        return found->buffer;
      };
      if ((staged || direct) && p.input_f32)
        throw MetalBackendError("16x16 projections require BF16 activations");
      if (buffer(0).sizeBytes() < (uint64_t(p.row_start) + p.rows) * p.input_size * (p.input_f32 ? 4 : 2) ||
          buffer(10).sizeBytes() < (uint64_t(p.row_start) + p.rows) * p.out_stride * (p.output_f32 ? 4 : 2) ||
          (p.epilogue && buffer(11).sizeBytes() < (uint64_t(p.row_start) + p.rows) * p.out_stride * 2))
        throw MetalBackendError("Vulkan projection input or output is out of bounds");
      if(native && ((uint64_t(p.row_start)+p.rows)*p.input_size*2>UINT32_MAX ||
                    (uint64_t(p.row_start)+p.rows)*p.out_stride*(p.output_f32?4:2)>UINT32_MAX))
        throw MetalBackendError("native matrix activation offsets exceed 32 bits");
      for (uint32_t i = 0; i < 3; ++i) {
        const uint32_t f = p.fmt[i];
        if (direct && p.cols[i] && f != GGUF_FMT_Q6K)
          throw MetalBackendError("direct 16x16 projections require Q6_K weights");
        if (f != 0xffffffffu && !supportedFormat(f)) throw MetalBackendError("Vulkan projection format is not ported");
        if (uint64_t(p.offset[i]) + p.cols[i] > p.out_stride)
          throw MetalBackendError("Vulkan projection segment is out of bounds");
        if (p.cols[i]) {
          if (f == 0xffffffffu) {
            if (buffer(1 + 3 * i).sizeBytes() < uint64_t(p.cols[i]) * p.plane_input_size * 4)
              throw MetalBackendError("Vulkan float projection weight is out of bounds");
          } else {
            const auto &format = kQuantFormats[f];
            const auto groups = p.plane_input_size / 32;
            if (groups % format.meta_groups) throw MetalBackendError("invalid Vulkan weight metadata geometry");
            const auto last = quant_tile_index(p.cols[i] - 1, groups - 1, groups) + 1;
            const auto metaLast = quant_tile_index(p.cols[i] - 1, groups / format.meta_groups - 1, groups / format.meta_groups) + 1;
            if(native && (last*format.plane0_bytes>UINT32_MAX || last*format.plane1_bytes>UINT32_MAX || metaLast*format.meta_bytes>UINT32_MAX))
              throw MetalBackendError("native quantized weight offsets exceed 32 bits");
            if (buffer(1 + 3 * i).sizeBytes() < last * format.plane0_bytes ||
                (format.plane1_bytes && buffer(2 + 3 * i).sizeBytes() < last * format.plane1_bytes) ||
                buffer(3 + 3 * i).sizeBytes() < metaLast * format.meta_bytes)
              throw MetalBackendError("Vulkan quantized projection weight is out of bounds");
          }
        }
        constants.push_back(f);
        key += ":f" + std::to_string(f);
      }
      for (uint32_t value : {p.epilogue, p.output_f32, p.input_size, p.plane_input_size, p.out_stride, uint32_t(p.rows % 8 == 0)}) {
        constants.push_back(value);
        key += ":" + std::to_string(value);
      }
      uint32_t segments = 0;
      bool allQ6=true;for(uint32_t i=0;i<3;++i)if(p.cols[i]&&p.fmt[i]!=GGUF_FMT_Q6K)allQ6=false;
      const uint32_t nativeTile=registered?16:capabilities.nativeMatrixDirectLayout?vulkanQ6MatrixTile(allQ6):8;
      if(staged&&dispatch.threadgroups.z!=1&&dispatch.threadgroups.z!=2&&dispatch.threadgroups.z!=4&&dispatch.threadgroups.z!=8)
        throw MetalBackendError("invalid staged projection split count");
      const uint32_t kSplits = staged?uint32_t(dispatch.threadgroups.z):registered?vulkanRegisterSplits(p.input_size,p.out_stride):!native && threads.x == 256 ? vulkanLinearKSplits(p.input_size) : 1;
      const uint32_t columnsPerGroup = direct ? 16 : staged ? 64 : native ? uint32_t(threads.x / 32) * nativeTile : threads.x == 256 ? 8 / kSplits : 16;
      const uint32_t rowTile = registered16 ? 16 : direct ? (p.rows<=8?8:16) : staged ? (p.rows<=8?8:p.rows<=16?16:32) : 8;
      if(registered){
        const uint64_t paddedRows=(uint64_t(p.rows)+rowTile-1)/rowTile*rowTile;
        if(p.input_f32||p.input_size%64||
           buffer(13).sizeBytes()<paddedRows*p.input_size*2||
           buffer(14).sizeBytes()<paddedRows*(p.input_size/16+p.input_size/32)*4||
           (kSplits>1&&buffer(15).sizeBytes()<uint64_t(p.rows)*p.out_stride*kSplits*4))
          throw MetalBackendError("invalid register projection workspace");
        if(registered16&&!capabilities.nativeMixedMatrix16Direct)
          throw MetalBackendError("native 16x16 register layout is not qualified");
      }
      if(staged&&kSplits>1&&buffer(15).sizeBytes()<uint64_t(p.rows)*p.out_stride*kSplits*4)
        throw MetalBackendError("invalid staged projection partial workspace");
      bool fullColumns = true;
      uint64_t tiles = 0;
      for (uint32_t columns : p.cols) {
        if (columns) ++segments;
        fullColumns = fullColumns && columns % columnsPerGroup == 0;
        tiles += (uint64_t(columns) + columnsPerGroup - 1) / columnsPerGroup;
      }
      if (dispatch.threadgroups.x != tiles || dispatch.threadgroups.y != (uint64_t(p.rows) + rowTile-1) / rowTile || dispatch.threadgroups.z != (registered||staged?kSplits:1) || (staged&&threads.x!=64) || (direct&&threads.x!=32))
        throw MetalBackendError("Vulkan projection grid does not match its geometry");
      constants.push_back(segments);
      constants.push_back(fullColumns);
      key += ":s" + std::to_string(segments) + ":c" + std::to_string(fullColumns);
      const bool alignedInput = buffer(0).gpuAddress() % 4 == 0;
      constants.push_back(alignedInput); constants.push_back(kSplits);
      const bool sharedAccumulators = std::getenv("SPLASH_VK_SHARED_ACCUM") && std::strcmp(std::getenv("SPLASH_VK_SHARED_ACCUM"), "1") == 0;
      constants.push_back(sharedAccumulators);
      const bool fourChains = std::getenv("SPLASH_VK_FOUR_CHAINS") && std::strcmp(std::getenv("SPLASH_VK_FOUR_CHAINS"), "1") == 0;
      constants.push_back(fourChains);
      constants.push_back(p.active_rows);
      constants.push_back(p.input_f32);
      constants.push_back(allQ6?16:32);
      constants.push_back(capabilities.nativeMatrixDirectLayout);
      constants.push_back(nativeTile);
      key+=":ntile"+std::to_string(nativeTile);
      if(registered16){
        const bool spread=std::getenv("SPLASH_VK_REGISTER_SPREAD")&&std::strcmp(std::getenv("SPLASH_VK_REGISTER_SPREAD"),"1")==0;
        constants.push_back(16);constants.push_back(1);constants.push_back(spread);key+=":spread"+std::to_string(spread);
      }
      if(registered&&!registered16){
        const auto counter=std::find_if(dispatch.buffers.begin(),dispatch.buffers.end(),[](const auto &b){return b.index==16;});
        const bool fused=counter!=dispatch.buffers.end();
        if(fused&&counter->buffer.sizeBytes()<tiles*((uint64_t(p.rows)+7)/8)*4)
          throw MetalBackendError("invalid fused reduction counter workspace");
        constants.push_back(fused);key+=":fused_reduce"+std::to_string(fused);
      }
      if(registered)source+="_f"+std::to_string(p.fmt[0]);
      if(registered&&std::any_of(dispatch.buffers.begin(),dispatch.buffers.end(),[](const auto &b){return b.index==17;})){source+=p.fmt[0]==GGUF_FMT_Q5K?"_q5":"_q6";key+=":native_layout_fmt"+std::to_string(p.fmt[0]);}

      if(staged||direct){constants.push_back(rowTile);constants.push_back(rowTile>16?2:1);key+=":rows"+std::to_string(rowTile);}
      key += ":a" + std::to_string(alignedInput) + ":k" + std::to_string(kSplits) + ":sacc" + std::to_string(sharedAccumulators) + ":chains" + std::to_string(fourChains) + ":active" + std::to_string(p.active_rows) + ":input32" + std::to_string(p.input_f32);
      if (!native && threads.x == 256) source = "vulkan_gguf_matvec";
    }
    if (source == "norm_rms" || source == "norm_rms_f32" ||
        source == "norm_rms_staged" || source == "norm_rms_staged_f32" ||
        source == "norm_rms_table16_decode" || source == "norm_rms_table16_decode_f32") {
      constants.push_back(source.ends_with("_f32"));
      constants.push_back(source.find("_staged") != std::string::npos);
      const bool table16=source.find("_table16")!=std::string::npos;
      constants.push_back(table16);
      const auto widthBytes=std::find_if(dispatch.bytes.begin(),dispatch.bytes.end(),[table16](const auto &b){return b.index==(table16?5u:3u)&&b.data&&b.sizeBytes==4;});
      if(widthBytes==dispatch.bytes.end())throw MetalBackendError("missing RMS normalization width");
      uint32_t width=0;std::memcpy(&width,widthBytes->data,4);
      if(!width)throw MetalBackendError("invalid RMS normalization width");
      constants.push_back(width);key+=":width"+std::to_string(width);
      source = "norm_rms";
    }
    if (source == "gguf_repack") {
      const auto bytes = std::find_if(dispatch.bytes.begin(), dispatch.bytes.end(), [](const auto &b) { return b.index == 2; });
      if (bytes == dispatch.bytes.end() || !bytes->data || bytes->sizeBytes != sizeof(GgufRepackParams))
        throw MetalBackendError("invalid GGUF repack parameters");
      GgufRepackParams p;
      std::memcpy(&p, bytes->data, sizeof(p));
      if (!supportedFormat(p.fmt))
        throw MetalBackendError("GGUF format is not ported: " + std::to_string(p.fmt));
      const auto &format = kQuantFormats[p.fmt];
      if (!p.rows || p.rows % QUANT_TILE_ROWS || !p.input_size || p.input_size % 256 ||
          p.src_row_bytes < uint64_t(p.input_size) / format.block_elements * format.block_bytes)
        throw MetalBackendError("invalid GGUF repack geometry");
      const auto input = std::find_if(dispatch.buffers.begin(), dispatch.buffers.end(), [](const auto &b) { return b.index == 0; });
      const auto output = std::find_if(dispatch.buffers.begin(), dispatch.buffers.end(), [](const auto &b) { return b.index == 1; });
      const uint64_t groups = uint64_t(p.rows) * (p.input_size / 32);
      if (input == dispatch.buffers.end() || output == dispatch.buffers.end() ||
          input->buffer.sizeBytes() < uint64_t(p.rows) * p.src_row_bytes ||
          output->buffer.sizeBytes() < groups * format.plane0_bytes ||
          (format.plane1_bytes && output->buffer.sizeBytes() < p.dst_plane1 + groups * format.plane1_bytes) ||
          output->buffer.sizeBytes() < p.dst_meta + groups / format.meta_groups * format.meta_bytes)
        throw MetalBackendError("GGUF repack buffer is out of bounds");
      constants.push_back(p.fmt);
      key += ":fmt" + std::to_string(p.fmt);
    }
    if(source=="vulkan_q6_repack8"||source=="vulkan_q6_repack16"){
      constants.push_back(source.ends_with("16"));key+=":wide"+std::to_string(constants.back());source="vulkan_q6_repack";
    }
    if (source == "vulkan_gguf_register_f3_q5") {
      const char *mode = std::getenv("SPLASH_VK_Q5_NARROW_LOADS");
      if (mode && std::strcmp(mode, "1") == 0) {
        source += "_narrow16";
        key += ":q5-narrow16";
      }
    }
    if(source=="draft_select_edges"){
      const char *mode=std::getenv("SPLASH_VK_METAL_EDGES");
      if(mode&&std::strcmp(mode,"1")==0){source+="_metal";key+=":metal_edges";}
    }
    if (auto found = pipelines.find(key); found != pipelines.end()) return found->second;
    const auto path = shaderDirectory / (source + ".spv");
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw MetalBackendError("Vulkan kernel not ported: " + path.string());
    const auto size = stream.tellg();
    if (size <= 0 || size % 4 != 0) throw MetalBackendError("invalid SPIR-V length: " + path.string());
    std::vector<uint32_t> code(static_cast<size_t>(size) / 4);
    stream.seekg(0);
    if (!stream.read(reinterpret_cast<char *>(code.data()), size))
      throw MetalBackendError("failed reading SPIR-V: " + path.string());
    VkShaderModuleCreateInfo moduleInfo{};
    moduleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleInfo.codeSize = static_cast<size_t>(size);
    moduleInfo.pCode = code.data();
    VkShaderModule module;
    checked(vkCreateShaderModule(device, &moduleInfo, nullptr, &module), "create shader module");
    std::vector<VkSpecializationMapEntry> entries;
    for (uint32_t i = 0; i < constants.size(); ++i) entries.push_back({i, i * 4, 4});
    VkSpecializationInfo specialization{uint32_t(entries.size()), entries.data(), constants.size() * 4, constants.data()};
    VkComputePipelineCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    info.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    info.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    info.stage.module = module;
    info.stage.pName = "main";
    info.stage.pSpecializationInfo = &specialization;
    info.layout = pipelineLayout;
    VkPipeline result;
    const auto status = vkCreateComputePipelines(device, pipelineCache, 1, &info, nullptr, &result);
    vkDestroyShaderModule(device, module, nullptr);
    checked(status, "create compute pipeline");
    pipelines.emplace(key, result);
    return result;
  }
};

struct Allocation {
  std::shared_ptr<Device> owner;
  VkBuffer buffer = VK_NULL_HANDLE;
  VkDeviceMemory memory = VK_NULL_HANDLE;
  VkDeviceAddress address = 0;
  bool preferUncached = false;
  void *mapped = nullptr;
  uint64_t bytes = 0;
  uint64_t committed = 0;
  BufferStorage storage = BufferStorage::Shared;

  void release() {
    if (!memory) return;
    if (mapped) vkUnmapMemory(owner->device, memory);
    vkDestroyBuffer(owner->device, buffer, nullptr);
    vkFreeMemory(owner->device, memory, nullptr);
    {
      std::lock_guard lock(owner->mutex);
      owner->stats.allocatedBytes -= committed;
      owner->stats.deviceCurrentAllocatedBytes = owner->stats.allocatedBytes;
    }
    mapped = nullptr;
    buffer = VK_NULL_HANDLE;
    memory = VK_NULL_HANDLE;
    address = 0;
    committed = 0;
  }

  void allocate() {
    VkBufferCreateInfo info{};
    info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size = std::max(uint64_t{4}, bytes);
    info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                 VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT;
    checked(vkCreateBuffer(owner->device, &info, nullptr, &buffer), "create buffer");
    VkMemoryRequirements requirements;
    vkGetBufferMemoryRequirements(owner->device, buffer, &requirements);
    uint32_t type = UINT32_MAX;
    for (uint32_t i = 0; i < owner->memory.memoryTypeCount; ++i) {
      const auto flags = owner->memory.memoryTypes[i].propertyFlags;
      if ((requirements.memoryTypeBits & (1u << i)) &&
          (flags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
              (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
        type = i;
        if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && (!preferUncached || !(flags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT))) break;
      }
    }
    if (type == UINT32_MAX) {
      vkDestroyBuffer(owner->device, buffer, nullptr);
      buffer = VK_NULL_HANDLE;
      throw MetalAllocationError("Vulkan coherent unified memory is unavailable");
    }
    VkMemoryAllocateFlagsInfo flags{};
    flags.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo allocation{};
    allocation.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocation.pNext = &flags;
    allocation.allocationSize = requirements.size;
    allocation.memoryTypeIndex = type;
    const auto result = vkAllocateMemory(owner->device, &allocation, nullptr, &memory);
    if (result != VK_SUCCESS) {
      vkDestroyBuffer(owner->device, buffer, nullptr);
      buffer = VK_NULL_HANDLE;
      throw MetalAllocationError("Vulkan allocation rejected: " + std::to_string(result));
    }
    committed = requirements.size;
    {
      std::lock_guard lock(owner->mutex);
      owner->stats.allocatedBytes += committed;
      owner->stats.peakAllocatedBytes = std::max(owner->stats.peakAllocatedBytes, owner->stats.allocatedBytes);
      owner->stats.deviceCurrentAllocatedBytes = owner->stats.allocatedBytes;
      owner->stats.devicePeakAllocatedBytes = owner->stats.peakAllocatedBytes;
    }
    checked(vkBindBufferMemory(owner->device, buffer, memory, 0), "bind buffer memory");
    checked(vkMapMemory(owner->device, memory, 0, VK_WHOLE_SIZE, 0, &mapped), "map buffer");
    VkBufferDeviceAddressInfo addressInfo{};
    addressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addressInfo.buffer = buffer;
    address = vkGetBufferDeviceAddress(owner->device, &addressInfo);
    if (!address) throw MetalBackendError("Vulkan returned a null GPU address");
  }

  ~Allocation() { release(); }
};

std::shared_ptr<Device> createDevice(const std::filesystem::path &shaders, double timeout) {
  auto result = std::make_shared<Device>();
  result->shaderDirectory = shaders;
  result->timeout = timeout;
  VkApplicationInfo application{};
  application.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  application.pApplicationName = "Splash Vulkan";
  application.apiVersion = g17gRequested()?VK_API_VERSION_1_3:VK_API_VERSION_1_2;
  VkInstanceCreateInfo instance{};
  instance.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instance.pApplicationInfo = &application;
  checked(vkCreateInstance(&instance, nullptr, &result->instance), "create instance");
  uint32_t count = 0;
  checked(vkEnumeratePhysicalDevices(result->instance, &count, nullptr), "enumerate GPUs");
  if (!count) throw MetalBackendError("no Vulkan device is available");
  std::vector<VkPhysicalDevice> devices(count);
  checked(vkEnumeratePhysicalDevices(result->instance, &count, devices.data()), "enumerate GPUs");
  for (auto physical : devices) {
    VkPhysicalDeviceProperties properties;
    vkGetPhysicalDeviceProperties(physical, &properties);
    if (properties.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) {
      result->physical = physical;
      break;
    }
  }
  if (!result->physical) throw MetalBackendError("a hardware Vulkan GPU is required");
  vkGetPhysicalDeviceProperties(result->physical, &result->properties);
  vkGetPhysicalDeviceMemoryProperties(result->physical, &result->memory);
  uint32_t familyCount;
  vkGetPhysicalDeviceQueueFamilyProperties(result->physical, &familyCount, nullptr);
  std::vector<VkQueueFamilyProperties> families(familyCount);
  vkGetPhysicalDeviceQueueFamilyProperties(result->physical, &familyCount, families.data());
  bool found = false;
  for (uint32_t i = 0; i < familyCount; ++i) {
    if (families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
      result->queueFamily = i;
      result->timestampBits = families[i].timestampValidBits;
      found = true;
      break;
    }
  }
  if (!found || !result->timestampBits) throw MetalBackendError("Vulkan compute timestamps are unavailable");
  uint32_t extensionCount = 0;
  checked(vkEnumerateDeviceExtensionProperties(result->physical, nullptr, &extensionCount, nullptr), "enumerate GPU extensions");
  std::vector<VkExtensionProperties> extensions(extensionCount);
  checked(vkEnumerateDeviceExtensionProperties(result->physical, nullptr, &extensionCount, extensions.data()), "enumerate GPU extensions");
  const bool hasCooperative = std::any_of(extensions.begin(),extensions.end(),[](const auto &e){return std::strcmp(e.extensionName,VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME)==0;});
  const bool hasBfloat = std::any_of(extensions.begin(),extensions.end(),[](const auto &e){return std::strcmp(e.extensionName,VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME)==0;});
  const bool hasShaderObject=std::any_of(extensions.begin(),extensions.end(),[](const auto&e){return std::strcmp(e.extensionName,VK_EXT_SHADER_OBJECT_EXTENSION_NAME)==0;});
  VkPhysicalDeviceShaderObjectFeaturesEXT objectFeatures{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};
  VkPhysicalDeviceVulkan13Features features13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
  VkPhysicalDeviceShaderBfloat16FeaturesKHR bfloatFeatures{};
  bfloatFeatures.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperativeFeatures{};
  cooperativeFeatures.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
  cooperativeFeatures.pNext=hasBfloat?&bfloatFeatures:nullptr;
  bfloatFeatures.pNext=hasShaderObject&&g17gRequested()?&objectFeatures:nullptr;
  VkPhysicalDeviceVulkan12Features features12{};
  features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  features13.pNext=hasCooperative?&cooperativeFeatures:nullptr;
  features12.pNext=g17gRequested()?static_cast<void*>(&features13):(hasCooperative?static_cast<void*>(&cooperativeFeatures):nullptr);
  VkPhysicalDeviceVulkan11Features features11{};
  features11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
  features11.pNext = &features12;
  VkPhysicalDeviceFeatures2 features{};
  features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
  features.pNext = &features11;
  vkGetPhysicalDeviceFeatures2(result->physical, &features);
  if (!features12.bufferDeviceAddress || !features12.timelineSemaphore ||
      !features11.storageBuffer16BitAccess || !features.features.shaderInt64 || !features.features.shaderInt16)
    throw MetalBackendError("Vulkan BDA, timeline semaphores and 16/64-bit storage are required");
  bool native8=false,native16=false,bfloat8=false,bfloat16=false;
  if(hasCooperative && cooperativeFeatures.cooperativeMatrix && features12.shaderFloat16 && features12.vulkanMemoryModel) {
    const auto getProperties=reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(vkGetInstanceProcAddr(result->instance,"vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
    if(getProperties){uint32_t n=0;checked(getProperties(result->physical,&n,nullptr),"query matrix shapes");
      std::vector<VkCooperativeMatrixPropertiesKHR> shapes(n);for(auto &x:shapes)x.sType=VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
      checked(getProperties(result->physical,&n,shapes.data()),"read matrix shapes");
      for(const auto &x:shapes)native8=native8||(x.MSize==8&&x.NSize==8&&x.KSize==8&&x.AType==VK_COMPONENT_TYPE_FLOAT16_KHR&&x.BType==VK_COMPONENT_TYPE_FLOAT16_KHR&&x.CType==VK_COMPONENT_TYPE_FLOAT32_KHR&&x.ResultType==VK_COMPONENT_TYPE_FLOAT32_KHR&&x.scope==VK_SCOPE_SUBGROUP_KHR&&!x.saturatingAccumulation);
      for(const auto &x:shapes)native16=native16||(hasBfloat&&bfloatFeatures.shaderBFloat16Type&&bfloatFeatures.shaderBFloat16CooperativeMatrix&&x.MSize==16&&x.NSize==16&&x.KSize==16&&x.AType==VK_COMPONENT_TYPE_BFLOAT16_KHR&&x.BType==VK_COMPONENT_TYPE_FLOAT16_KHR&&x.CType==VK_COMPONENT_TYPE_FLOAT32_KHR&&x.ResultType==VK_COMPONENT_TYPE_FLOAT32_KHR&&x.scope==VK_SCOPE_SUBGROUP_KHR&&!x.saturatingAccumulation);      for(const auto &x:shapes){
        const bool supported=hasBfloat&&bfloatFeatures.shaderBFloat16Type&&bfloatFeatures.shaderBFloat16CooperativeMatrix&&x.AType==VK_COMPONENT_TYPE_BFLOAT16_KHR&&x.BType==VK_COMPONENT_TYPE_BFLOAT16_KHR&&x.CType==VK_COMPONENT_TYPE_FLOAT32_KHR&&x.ResultType==VK_COMPONENT_TYPE_FLOAT32_KHR&&x.scope==VK_SCOPE_SUBGROUP_KHR&&!x.saturatingAccumulation;
        bfloat8=bfloat8||(supported&&x.MSize==8&&x.NSize==8&&x.KSize==8);
        bfloat16=bfloat16||(supported&&x.MSize==16&&x.NSize==16&&x.KSize==16);
      }

    }
  }
  result->nativeBfloatMatrix16=bfloat16;
  const bool enableBfloat=native16||bfloat8||bfloat16;
  const bool enableCooperative=native8||native16||bfloat8||bfloat16;
  const bool enableObjects=g17gRequested()&&hasShaderObject&&objectFeatures.shaderObject&&features13.dynamicRendering&&native16;
  result->shaderObjectsEnabled=enableObjects;
  VkPhysicalDeviceShaderObjectFeaturesEXT enabledObjects{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT};enabledObjects.shaderObject=enableObjects;
  VkPhysicalDeviceVulkan13Features enabled13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};enabled13.dynamicRendering=enableObjects;
  VkPhysicalDeviceCooperativeMatrixFeaturesKHR enabledCooperative{};
  enabledCooperative.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
  enabledCooperative.cooperativeMatrix=enableCooperative;
  VkPhysicalDeviceShaderBfloat16FeaturesKHR enabledBfloat{};
  enabledBfloat.sType=VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_BFLOAT16_FEATURES_KHR;
  enabledBfloat.shaderBFloat16Type=enableBfloat;
  enabledBfloat.shaderBFloat16CooperativeMatrix=enableBfloat;
  enabledCooperative.pNext=enableBfloat?&enabledBfloat:nullptr;
  enabledBfloat.pNext=enableObjects?&enabledObjects:nullptr;
  VkPhysicalDeviceVulkan12Features enabled12{};
  enabled12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  enabled12.bufferDeviceAddress = true;
  enabled12.timelineSemaphore = true;
  enabled12.shaderFloat16 = features12.shaderFloat16;
  enabled12.vulkanMemoryModel=enableCooperative;
  enabled13.pNext=enableCooperative?&enabledCooperative:nullptr;
  enabled12.pNext=enableObjects?static_cast<void*>(&enabled13):(enableCooperative?static_cast<void*>(&enabledCooperative):nullptr);
  VkPhysicalDeviceVulkan11Features enabled11{};
  enabled11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
  enabled11.storageBuffer16BitAccess = true;
  enabled11.pNext = &enabled12;
  VkPhysicalDeviceFeatures enabled{};
  enabled.shaderInt64 = true;
  enabled.shaderInt16 = true;
  const float priority = 1.0f;
  VkDeviceQueueCreateInfo queue{};
  queue.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue.queueFamilyIndex = result->queueFamily;
  queue.queueCount = 1;
  queue.pQueuePriorities = &priority;
  VkDeviceCreateInfo device{};
  device.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device.pNext = &enabled11;
  device.pEnabledFeatures = &enabled;
  device.queueCreateInfoCount = 1;
  device.pQueueCreateInfos = &queue;
  std::vector<const char *> deviceExtensions;
  if(enableCooperative)deviceExtensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
  if(enableBfloat)deviceExtensions.push_back(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME);
  if(enableObjects)deviceExtensions.push_back(VK_EXT_SHADER_OBJECT_EXTENSION_NAME);
  device.enabledExtensionCount=deviceExtensions.size();device.ppEnabledExtensionNames=deviceExtensions.data();
  checked(vkCreateDevice(result->physical, &device, nullptr, &result->device), "create device");
  vkGetDeviceQueue(result->device, result->queueFamily, 0, &result->queue);
  VkDescriptorSetLayoutBinding binding{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
  VkDescriptorSetLayoutCreateInfo descriptor{};
  descriptor.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
  descriptor.bindingCount = 1;
  descriptor.pBindings = &binding;
  checked(vkCreateDescriptorSetLayout(result->device, &descriptor, nullptr, &result->descriptorLayout), "create descriptor layout");
  VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t)};
  VkPipelineLayoutCreateInfo layout{};
  layout.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
  layout.setLayoutCount = 1;
  layout.pSetLayouts = &result->descriptorLayout;
  layout.pushConstantRangeCount = 1;
  layout.pPushConstantRanges = &push;
  checked(vkCreatePipelineLayout(result->device, &layout, nullptr, &result->pipelineLayout), "create pipeline layout");
  VkPipelineCacheCreateInfo cache{};
  cache.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
  checked(vkCreatePipelineCache(result->device, &cache, nullptr, &result->pipelineCache), "create pipeline cache");
  result->loadNativePairs();
  result->loadNativeSingles();
  auto &caps = result->capabilities;
  caps.backendApi = BackendApi::Vulkan;
  caps.nativeMatrixSupported=native8;caps.nativeBfloatMatrix8=bfloat8;caps.nativeBfloatMatrix16=bfloat16;
  caps.nativeMixedMatrix16=native16;
  caps.deviceName = result->properties.deviceName;
  caps.vulkanApiVersion = result->properties.apiVersion;
  caps.bufferDeviceAddress = true;
  caps.storageBuffer16BitAccess = true;
  caps.physicalMemoryBytes = uint64_t(sysconf(_SC_PHYS_PAGES)) * uint64_t(sysconf(_SC_PAGESIZE));
  caps.maxThreadgroupMemoryBytes = result->properties.limits.maxComputeSharedMemorySize;
  caps.maxThreadgroupWidth = std::min(result->properties.limits.maxComputeWorkGroupSize[0], result->properties.limits.maxComputeWorkGroupInvocations);
  for (uint32_t i = 0; i < result->memory.memoryTypeCount; ++i) {
    const auto flags = result->memory.memoryTypes[i].propertyFlags;
    if ((flags & (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) ==
        (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) caps.hasUnifiedMemory = true;
  }
  VkPhysicalDeviceMaintenance3Properties maintenance{};
  maintenance.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES;
  VkPhysicalDeviceSubgroupProperties subgroup{};
  subgroup.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
  maintenance.pNext = &subgroup;
  VkPhysicalDeviceProperties2 properties{};
  properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
  properties.pNext = &maintenance;
  vkGetPhysicalDeviceProperties2(result->physical, &properties);
  caps.subgroupSize = subgroup.subgroupSize;
  caps.maxBufferLengthBytes = maintenance.maxMemoryAllocationSize;
  uint64_t localBytes = 0;
  const bool hasBudget = std::any_of(extensions.begin(), extensions.end(), [](const auto &extension) {
    return std::strcmp(extension.extensionName, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME) == 0;
  });
  VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{};
  budget.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT;
  if (hasBudget) {
    VkPhysicalDeviceMemoryProperties2 memory{};
    memory.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2;
    memory.pNext = &budget;
    vkGetPhysicalDeviceMemoryProperties2(result->physical, &memory);
  }
  for (uint32_t i = 0; i < result->memory.memoryHeapCount; ++i)
    if (result->memory.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
      localBytes = std::max(localBytes, budget.heapBudget[i] ? budget.heapBudget[i] : result->memory.memoryHeaps[i].size);
  caps.recommendedMaxWorkingSetBytes = std::min(localBytes, caps.physicalMemoryBytes);
  return result;
}

} // namespace

struct MetalBuffer::Impl {
  std::shared_ptr<Allocation> allocation;
  uint64_t offset = 0;
  uint64_t bytes = 0;
};

MetalBuffer::MetalBuffer() = default;
MetalBuffer::~MetalBuffer() = default;
MetalBuffer::MetalBuffer(const MetalBuffer &) = default;
MetalBuffer &MetalBuffer::operator=(const MetalBuffer &) = default;
MetalBuffer::MetalBuffer(MetalBuffer &&) noexcept = default;
MetalBuffer &MetalBuffer::operator=(MetalBuffer &&) noexcept = default;
MetalBuffer::MetalBuffer(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
MetalBuffer::operator bool() const noexcept { return bool(impl_); }
uint64_t MetalBuffer::sizeBytes() const noexcept { return impl_ ? impl_->bytes : 0; }
uint64_t MetalBuffer::allocatedBytes() const noexcept { return impl_ ? impl_->allocation->committed : 0; }
BufferStorage MetalBuffer::storage() const noexcept { return impl_ ? impl_->allocation->storage : BufferStorage::Shared; }
void *MetalBuffer::contents() const noexcept {
  if (!impl_ || storage() != BufferStorage::Shared || !impl_->allocation->mapped) return nullptr;
  return static_cast<std::byte *>(impl_->allocation->mapped) + impl_->offset;
}
uint64_t MetalBuffer::gpuAddress() const noexcept {
  return impl_ && impl_->allocation->address ? impl_->allocation->address + impl_->offset : 0;
}
bool MetalBuffer::sameView(const MetalBuffer &other) const noexcept {
  if (!impl_ || !other.impl_) return !impl_ && !other.impl_;
  return impl_->allocation == other.impl_->allocation && impl_->offset == other.impl_->offset && impl_->bytes == other.impl_->bytes;
}

struct SharedEvent::Impl {
  std::shared_ptr<Device> owner;
  VkSemaphore semaphore = VK_NULL_HANDLE;
  ~Impl() { if (semaphore) vkDestroySemaphore(owner->device, semaphore, nullptr); }
};

SharedEvent::SharedEvent() = default;
SharedEvent::~SharedEvent() = default;
SharedEvent::SharedEvent(const SharedEvent &) = default;
SharedEvent &SharedEvent::operator=(const SharedEvent &) = default;
SharedEvent::SharedEvent(SharedEvent &&) noexcept = default;
SharedEvent &SharedEvent::operator=(SharedEvent &&) noexcept = default;
SharedEvent::SharedEvent(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
SharedEvent::operator bool() const noexcept { return bool(impl_); }
void *SharedEvent::nativeHandle() const noexcept { return impl_ ? reinterpret_cast<void *>(impl_->semaphore) : nullptr; }
void SharedEvent::signal(uint64_t value) const noexcept {
  if (!impl_) return;
  std::lock_guard lock(impl_->owner->queueMutex);
  uint64_t current = 0;
  auto result = vkGetSemaphoreCounterValue(impl_->owner->device, impl_->semaphore, &current);
  if (result == VK_SUCCESS && current < value) {
    VkSemaphoreSignalInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
    info.semaphore = impl_->semaphore;
    info.value = value;
    result = vkSignalSemaphore(impl_->owner->device, &info);
  }
  if (result != VK_SUCCESS) impl_->owner->fail("Vulkan event signal failed");
}
void SharedEvent::notify(uint64_t value, std::function<void()> callback) const {
  if (!impl_ || !callback) throw MetalBackendError("invalid Vulkan event notification");
  std::thread([impl = impl_, value, callback = std::move(callback)] {
    VkSemaphoreWaitInfo info{};
    info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    info.semaphoreCount = 1;
    info.pSemaphores = &impl->semaphore;
    info.pValues = &value;
    const auto result = vkWaitSemaphores(impl->owner->device, &info, UINT64_MAX);
    if (result != VK_SUCCESS) { impl->owner->fail("Vulkan event notification failed"); return; }
    std::lock_guard lock(impl->owner->callbackMutex);
    try { callback(); } catch (...) { impl->owner->fail("Vulkan event callback threw"); }
  }).detach();
}

struct PackedCacheEntry {
 uint64_t address=0,bytes=0; std::weak_ptr<Allocation> source; std::atomic<bool> ready=false;
};

struct CommandTicket::State {
  std::shared_ptr<Device> owner;
  uint64_t generation = 0;
  VkCommandPool pool = VK_NULL_HANDLE;
  VkDescriptorPool descriptors = VK_NULL_HANDLE;
  VkFence fence = VK_NULL_HANDLE;
  VkQueryPool queries = VK_NULL_HANDLE;
  std::vector<MetalBuffer> resources;
  std::vector<std::shared_ptr<Allocation>> resident;
  MetalBuffer checkedCounters;
  std::vector<std::shared_ptr<PackedCacheEntry>> preparedCaches;
  std::string profilePath;
  std::vector<std::string> profileNames;
  std::mutex mutex;
  std::condition_variable condition;
  bool done = false;
  bool consumed = false;
  std::string error;
  CommandTiming timing;
  Clock::time_point start;
  Clock::time_point submitted;

  ~State() {
    if (queries) vkDestroyQueryPool(owner->device, queries, nullptr);
    if (fence) vkDestroyFence(owner->device, fence, nullptr);
    if (descriptors) vkDestroyDescriptorPool(owner->device, descriptors, nullptr);
    if (pool) vkDestroyCommandPool(owner->device, pool, nullptr);
  }

  CommandTiming wait() {
    std::unique_lock lock(mutex);
    while (!done) {
      condition.wait_for(lock, std::chrono::seconds(1));
      if (!done && (std::chrono::duration<double>(Clock::now() - submitted).count() > owner->timeout ||
                    (owner->waitInterrupt && owner->waitInterrupt()))) {
        owner->fail("Vulkan command wait interrupted or timed out");
        throw MetalBackendError("Vulkan command wait interrupted or timed out");
      }
    }
    if (!consumed) {
      consumed = true;
      if (owner->generation.load() == generation) owner->inFlight = false;
    }
    if (!error.empty()) throw MetalBackendError(error);
    return timing;
  }
};

CommandTicket::CommandTicket() = default;
CommandTicket::CommandTicket(std::shared_ptr<State> state) : state_(std::move(state)) {}
CommandTicket::~CommandTicket() { if (state_) { try { state_->wait(); } catch (...) {} } }
CommandTicket::CommandTicket(CommandTicket &&) noexcept = default;
CommandTicket &CommandTicket::operator=(CommandTicket &&other) noexcept {
  if (this != &other) {
    if (state_) { try { state_->wait(); } catch (...) {} }
    state_ = std::move(other.state_);
  }
  return *this;
}
bool CommandTicket::ready() const noexcept {
  if (!state_) return true;
  std::lock_guard lock(state_->mutex);
  return state_->done;
}
CommandTiming CommandTicket::wait() { return state_ ? state_->wait() : CommandTiming{}; }

struct MetalBackend::Impl {
  std::shared_ptr<Device> device;
  std::mutex workspaceMutex;
  std::map<std::array<uint64_t,8>,std::shared_ptr<PackedCacheEntry>> packedCaches;
  std::map<std::pair<uint64_t,uint64_t>,std::weak_ptr<Allocation>> canonicalPairViews;
  std::map<std::pair<uint32_t,uint32_t>,MetalBuffer> nativePairParams;
  MetalBuffer nativeSinglePartials,nativeSingleCounters;
  std::map<std::array<uint32_t,3>,MetalBuffer> nativeSingleParams;
  MetalBuffer registerTable,registerSums,registerPartials,registerCounters,registerCacheScratch;
};

DeviceCapabilities probeDeviceCapabilities() { return createDevice({}, kCommandTimeoutSeconds)->capabilities; }
MetalBackend::MetalBackend(std::string shaders, double, double timeout) : impl_(std::make_unique<Impl>()) {
  if (!(timeout > 0)) throw MetalBackendError("invalid command timeout");
  impl_->device = createDevice(shaders, timeout);
  const bool matrix16Requested=g17gRequested() || (std::getenv("SPLASH_VK_METAL_STAGED") && std::strcmp(std::getenv("SPLASH_VK_METAL_STAGED"),"1")==0) || (std::getenv("SPLASH_VK_DIRECT_MATRIX") && std::strcmp(std::getenv("SPLASH_VK_DIRECT_MATRIX"),"1")==0) || (std::getenv("SPLASH_VK_REGISTER") && std::strcmp(std::getenv("SPLASH_VK_REGISTER"),"16")==0);
  if(matrix16Requested&&!impl_->device->capabilities.nativeMixedMatrix16)throw MetalBackendError("requested native 16x16 mixed matrix feature is unavailable");
  if(matrix16Requested){
    auto probe=allocateBuffer(2048,BufferStorage::Shared,"16x16 matrix fragment qualification");
    std::memset(probe.contents(),0,2048);
    (void)submit({"vulkan_coop_layout16",{{0,view(probe,0,512)},{1,view(probe,512,512)},{2,view(probe,1024,1024)}},{},{1,1,1},{32,1,1}});
    const auto *a=static_cast<const uint16_t*>(probe.contents());const auto *b=a+256;
    const auto *c=reinterpret_cast<const float*>(a+512);
    bool known=true;
    for(uint32_t lane=0;lane<32;++lane)for(uint32_t i=0;i<8;++i){
      const uint32_t row=(lane&6)|((lane>>1)&8)|(i>>2);
      const uint32_t ca=((lane>>3)&1)|((lane&1)<<3)|((i&3)<<1),cb=((lane&1)<<2)|(lane&8)|(i&3);
      const float value=float(lane*8+i);const uint16_t abits=uint16_t(std::bit_cast<uint32_t>(value)>>16);
      known=known && a[row*16+ca]==abits && c[row*16+cb]==value;
      const uint16_t hbits=value==0?0:uint16_t(((std::bit_cast<uint32_t>(value)>>13)&0x3ffu)|((((std::bit_cast<uint32_t>(value)>>23)&255u)-112u)<<10));
      known=known && b[row*16+cb]==hbits;
    }
    impl_->device->capabilities.nativeMixedMatrix16Direct=known;
    std::fprintf(stderr,"Vulkan 16x16 mixed fragment layout: %s\n",known?"qualified":"unrecognized");
    if(!known)throw MetalBackendError("16x16 mixed matrix fragment layout does not match Metal port");
  }
  if(impl_->device->capabilities.nativeMatrixSupported && std::getenv("SPLASH_VK_NATIVE_MATRIX") && std::strcmp(std::getenv("SPLASH_VK_NATIVE_MATRIX"),"1")==0){
    auto probe=allocateBuffer(512,BufferStorage::Shared,"matrix fragment qualification");
    std::memset(probe.contents(),0,512);
    (void)submit({"vulkan_coop_layout",{{0,view(probe,0,128)},{1,view(probe,128,128)},{2,view(probe,256,256)}},{},{1,1,1},{32,1,1}});
    const auto *a=static_cast<const uint16_t*>(probe.contents());const auto *b=a+64;
    const auto *c=reinterpret_cast<const float*>(a+128);bool known=true;
    for(uint32_t lane=0;lane<32;++lane)for(uint32_t i=0;i<2;++i){
      const uint32_t row=((lane>>1)&3)|((lane>>2)&4),col=((lane&1)<<1)|((lane>>1)&4)|i,index=row*8+col;
      const float value=float(lane*2+i);const uint32_t bits=std::bit_cast<uint32_t>(value);
      const uint16_t half=value==0?0:uint16_t((((bits>>23)-112)<<10)|((bits>>13)&1023));
      known=known&&a[index]==half&&b[index]==half&&c[index]==value;
    }
    impl_->device->capabilities.nativeMatrixDirectLayout=known;
    std::fprintf(stderr,"Vulkan 8x8 direct fragment layout: %s\n",known?"qualified":"generic loads");
  }
}
MetalBackend::~MetalBackend() { stop(); }
void MetalBackend::setOperationGuard(std::function<void()> guard) { impl_->device->operationGuard = std::move(guard); }
void MetalBackend::setWaitInterrupt(std::function<bool()> interrupt) { impl_->device->waitInterrupt = std::move(interrupt); }
void MetalBackend::checkOperation() const { impl_->device->check(); }
void MetalBackend::stop() noexcept { impl_->device->stopped = true; }
const DeviceCapabilities &MetalBackend::capabilities() const noexcept { return impl_->device->capabilities; }

MetalBuffer MetalBackend::allocateBuffer(uint64_t bytes, BufferStorage storage, std::string_view label) {
  auto device = impl_->device;
  device->check();
  auto allocation = std::make_shared<Allocation>();
  allocation->owner = device;
  allocation->bytes = bytes;
  allocation->storage = storage;
  allocation->preferUncached = bytes >= (1u<<20) && !label.starts_with("load/") && std::getenv("SPLASH_VK_UNCACHED_GPU") && std::strcmp(std::getenv("SPLASH_VK_UNCACHED_GPU"),"1")==0;
  allocation->allocate();
  auto buffer = std::make_shared<MetalBuffer::Impl>();
  buffer->allocation = allocation;
  buffer->bytes = bytes;
  {
    std::lock_guard lock(device->mutex);
    std::erase_if(device->allocations, [](const auto &entry) { return entry.expired(); });
    device->allocations.push_back(allocation);
  }
  return MetalBuffer(std::move(buffer));
}
MetalBuffer MetalBackend::wrapSharedMemory(void *, uint64_t, std::shared_ptr<void>, std::string_view) {
  throw MetalBackendError("external host-memory import has not been ported; Linux ANE is disabled");
}
MetalBuffer MetalBackend::view(const MetalBuffer &base, uint64_t offset, uint64_t bytes) const {
  if (!base.impl_ || base.impl_->allocation->owner != impl_->device || offset > base.sizeBytes() || bytes > base.sizeBytes() - offset)
    throw MetalBackendError("invalid Vulkan buffer view");
  auto buffer = std::make_shared<MetalBuffer::Impl>();
  buffer->allocation = base.impl_->allocation;
  buffer->offset = base.impl_->offset + offset;
  buffer->bytes = bytes;
  return MetalBuffer(std::move(buffer));
}
SharedEvent MetalBackend::newSharedEvent() {
  auto event = std::make_shared<SharedEvent::Impl>();
  event->owner = impl_->device;
  VkSemaphoreTypeCreateInfo type{};
  type.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  type.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  VkSemaphoreCreateInfo info{};
  info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  info.pNext = &type;
  checked(vkCreateSemaphore(event->owner->device, &info, nullptr, &event->semaphore), "create shared event");
  return SharedEvent(std::move(event));
}
void MetalBackend::releaseMemory(const MetalBuffer &buffer) {
  checkOperation();
  if (commandInFlight() || !buffer.impl_ || buffer.impl_->allocation->owner != impl_->device)
    throw MetalBackendError("unsafe Vulkan memory release");
  {
    std::lock_guard lock(impl_->workspaceMutex);
    std::erase_if(impl_->packedCaches,[&](const auto &item){auto source=item.second->source.lock();return !source||source==buffer.impl_->allocation;});
    std::erase_if(impl_->canonicalPairViews,[&](const auto&item){auto source=item.second.lock();return !source||source==buffer.impl_->allocation;});impl_->registerCacheScratch={};impl_->registerTable={};impl_->registerSums={};impl_->registerPartials={};impl_->registerCounters={};
  }
  buffer.impl_->allocation->release();
}
void MetalBackend::restoreMemory(const MetalBuffer &buffer) {
  checkOperation();
  if (commandInFlight() || !buffer.impl_ || buffer.impl_->allocation->owner != impl_->device)
    throw MetalBackendError("unsafe Vulkan memory restore");
  if (!buffer.impl_->allocation->memory) buffer.impl_->allocation->allocate();
}

CommandTiming MetalBackend::submit(const ComputeDispatch &dispatch) {
  return submitCommandAsync(std::span(&dispatch, 1)).wait();
}
CommandTicket MetalBackend::submitCommandAsync(std::span<const ComputeDispatch> dispatches, CommandCompletion completion) {
  return submitCommandAsync(Command{dispatches, {}}, std::move(completion));
}
CommandTicket MetalBackend::submitCommandAsync(const Command &command, CommandCompletion completion) {
  auto device = impl_->device;
  device->check();
  if (!command.events.empty()) throw MetalBackendError("Vulkan command event encoding is not implemented yet");
  const bool nativePairMode=g17gPairRequested()&&device->nativePairsReady&&device->capabilities.nativeMixedMatrix16Direct;
  const bool nativeSingleMode=g17gSingleRequested()&&device->nativeSinglesReady&&device->capabilities.nativeMixedMatrix16Direct;
  if(nativePairMode||nativeSingleMode) {
    const auto parameters=[](const ComputeDispatch&d,VulkanLinearParams&p){
      if(d.pipelineName!="vulkan_gguf_coop"||d.indirectThreadgroups.buffer||d.threadgroups.z!=1)return false;
      for(const auto&b:d.bytes)if(b.index==12&&b.data&&b.sizeBytes==sizeof(p)){std::memcpy(&p,b.data,sizeof(p));return true;}
      return false;
    };
    const auto get=[](const ComputeDispatch&d,uint32_t slot)->const MetalBuffer*{for(const auto&b:d.buffers)if(b.index==slot)return &b.buffer;return nullptr;};
    const auto overlap=[](const MetalBuffer&a,const MetalBuffer&b){return a.gpuAddress()<b.gpuAddress()+b.sizeBytes()&&b.gpuAddress()<a.gpuAddress()+a.sizeBytes();};
    {
      std::lock_guard lock(impl_->workspaceMutex);
      std::erase_if(impl_->canonicalPairViews,[](const auto&v){return v.second.expired();});
      if(!impl_->canonicalPairViews.empty()){
        const auto first=impl_->canonicalPairViews.begin()->first.first;
        const auto&last=*impl_->canonicalPairViews.rbegin();const auto limit=last.first.first+last.first.second;
        for(const auto&d:command.dispatches)if(d.pipelineName.starts_with("vulkan_gguf_")||d.pipelineName.starts_with("vulkan_register_reduce"))
          for(const auto&b:d.buffers)if((b.index==0||b.index==10||b.index==11)&&b.buffer.impl_&&b.buffer.sizeBytes()){
            auto begin=b.buffer.gpuAddress(),end=begin+b.buffer.sizeBytes();if(end<=first||begin>=limit)continue;
            for(const auto&[v,owner]:impl_->canonicalPairViews)if(owner.lock()==b.buffer.impl_->allocation&&begin<v.first+v.second&&v.first<end)
              throw MetalBackendError("kernel cannot use immutable quantized weight view as activation/output: "+d.pipelineName);
          }
      }
    }
    struct Producer {size_t index;VulkanLinearParams params;};
    using Key=std::array<uint64_t,2>;
    std::map<Key,Producer> producers;
    std::map<size_t,size_t> nativeAt;
    unsigned canonicalPairs=0;
    for(size_t i=0;i<command.dispatches.size();++i){
      const auto&d=command.dispatches[i];VulkanLinearParams p{};
      if(d.pipelineName.starts_with("vulkan_gguf_pair_g17g_f")){const auto*written=get(d,11);if(written)producers.erase(Key{written->gpuAddress(),0});continue;}
      if(!parameters(d,p)||!p.rows||!p.input_size||p.input_size%256||p.plane_input_size!=p.input_size||!p.out_stride||p.out_stride%32||p.cols[0]!=p.out_stride||p.cols[1]||p.cols[2]||p.offset[0])continue;
      const auto*x=get(d,0);const auto*go=get(d,p.epilogue==2?11:10);if(!x||!go)continue;
      const Key key{go->gpuAddress(),p.row_start};
      if(p.epilogue==0){producers[key]={i,p};continue;}
      if(p.epilogue!=2||(p.fmt[0]!=3&&p.fmt[0]!=4))continue;
      auto producer=producers.find(key);if(producer==producers.end())continue;
      const auto&g=command.dispatches[producer->second.index];const auto&gp=producer->second.params;
      const auto*gx=get(g,0);const auto*gout=get(g,10);const auto*out=get(d,10);
      bool match=gp.fmt[0]==p.fmt[0]&&gp.input_size==p.input_size&&gp.plane_input_size==p.plane_input_size&&gp.out_stride==p.out_stride&&gp.rows==p.rows&&gp.active_rows==p.active_rows&&gp.active_rows>=1&&gp.active_rows<=8&&gx&&gout&&out&&gx->sameView(*x)&&gout->sameView(*go);
      bool safe=match&&!overlap(*out,*x)&&!overlap(*go,*x)&&!overlap(*out,*go);
      for(uint32_t slot=1;slot<=3&&match;++slot){const auto*a=get(g,slot),*b=get(d,slot);match=a&&b;if(match)safe=safe&&!overlap(*out,*a)&&!overlap(*go,*a)&&!overlap(*out,*b)&&!overlap(*go,*b);}
      if(!match)continue;
      {
        std::lock_guard lock(impl_->workspaceMutex);
        for(const auto*projection:{&g,&d})for(uint32_t slot=1;slot<=3;++slot){
          const auto*w=get(*projection,slot);const auto begin=w->gpuAddress(),end=begin+w->sizeBytes();
          for(const auto&[key,entry]:impl_->packedCaches)if(entry->source.lock()==w->impl_->allocation&&begin<entry->address+entry->bytes&&entry->address<end)
            throw MetalBackendError("native pair requires canonical weights from a fresh process");
          impl_->canonicalPairViews[{begin,w->sizeBytes()}]=w->impl_->allocation;
        }
      }
      ++canonicalPairs;
      if(nativePairMode&&safe&&!gp.input_f32&&!p.input_f32&&!gp.output_f32&&!p.output_f32&&producer->second.index+1==i&&p.rows==8&&p.active_rows==8&&p.row_start==0)nativeAt[producer->second.index]=i;
    }
    if(!nativeAt.empty()){
      std::vector<ComputeDispatch> merged;merged.reserve(command.dispatches.size()-nativeAt.size());
      std::deque<std::array<uint32_t,52>> roots;
      for(size_t i=0;i<command.dispatches.size();++i){
        auto found=nativeAt.find(i);if(found==nativeAt.end()){merged.push_back(command.dispatches[i]);continue;}
        const auto&g=command.dispatches[i];const auto&up=command.dispatches[found->second];VulkanLinearParams p{};if(!parameters(g,p))throw MetalBackendError("native pair producer disappeared");
        MetalBuffer nativeParams;
        {
          std::lock_guard lock(impl_->workspaceMutex);
          auto key=std::make_pair(p.input_size,p.out_stride);auto entry=impl_->nativePairParams.find(key);
          if(entry==impl_->nativePairParams.end()){
            auto buffer=allocateBuffer(20,BufferStorage::Shared,"native pair runtime parameters");const uint32_t words[5]={p.input_size,1,p.out_stride,0,1};std::memcpy(buffer.contents(),words,20);entry=impl_->nativePairParams.emplace(key,std::move(buffer)).first;
          }
          nativeParams=entry->second;
        }
        auto d=g;d.pipelineName="vulkan_gguf_pair_g17g_f"+std::to_string(p.fmt[0]);d.buffers.clear();d.buffers.push_back({0,*get(g,0)});
        for(uint32_t slot=1;slot<=3;++slot){d.buffers.push_back({slot,*get(g,slot)});d.buffers.push_back({slot+3,*get(up,slot)});}
        d.buffers.push_back({10,*get(up,10)});d.buffers.push_back({11,*get(g,10)});d.buffers.push_back({20,nativeParams});
        roots.emplace_back();auto&root=roots.back();const uint64_t pointers[11]={get(g,0)->gpuAddress(),get(g,1)->gpuAddress(),get(g,2)->gpuAddress(),get(g,3)->gpuAddress(),get(up,1)->gpuAddress(),get(up,2)->gpuAddress(),get(up,3)->gpuAddress(),get(up,10)->gpuAddress(),get(g,10)->gpuAddress(),get(g,0)->gpuAddress(),nativeParams.gpuAddress()};
        for(uint32_t j=0;j<11;++j){root[2*j]=uint32_t(pointers[j]);root[2*j+1]=uint32_t(pointers[j]>>32);}
        root[34]=p.out_stride;std::memcpy(root.data()+35,p.fmt[0]==3?kNativeQ5Literals:kNativeQ6Literals,52);root[48]=uint32_t(pointers[0]);root[49]=uint32_t(pointers[0]>>32);
        std::erase_if(d.bytes,[](const auto&b){return b.index!=12;});d.bytes.push_back({22,root.data(),208});d.threadgroups={p.out_stride/32,1,1};d.threadsPerThreadgroup={32,1,1};merged.push_back(std::move(d));++i;
      }
      if(std::getenv("SPLASH_VK_G17G_PAIR_TRACE"))std::fprintf(stderr,"g17g_pair_native_pairs %zu canonical_pairs %u\n",nativeAt.size(),canonicalPairs);
      return submitCommandAsync(Command{merged,{}},std::move(completion));
    }
    if(canonicalPairs&&std::getenv("SPLASH_VK_G17G_PAIR_TRACE"))std::fprintf(stderr,"g17g_pair_native_pairs 0 canonical_pairs %u\n",canonicalPairs);
  }
  if(nativeSingleMode){
    const auto get=[](const ComputeDispatch&d,uint32_t slot)->const MetalBuffer*{for(const auto&b:d.buffers)if(b.index==slot)return &b.buffer;return nullptr;};
    const auto parameters=[](const ComputeDispatch&d,VulkanLinearParams&p){
      if(d.pipelineName!="vulkan_gguf_coop"||d.indirectThreadgroups.buffer||d.threadgroups.z!=1)return false;
      for(const auto&b:d.bytes)if(b.index==12&&b.data&&b.sizeBytes==sizeof(p)){
        std::memcpy(&p,b.data,sizeof(p));
        return p.input_size&&p.input_size%256==0&&p.plane_input_size==p.input_size&&p.out_stride&&p.out_stride%64==0&&p.cols[0]==p.out_stride&&!p.cols[1]&&!p.cols[2]&&!p.offset[0]&&(p.fmt[0]==3||p.fmt[0]==4)&&p.epilogue<=2&&(!p.output_f32||!p.epilogue);
      }
      return false;
    };
    const auto splitsFor=[](const VulkanLinearParams&p){return p.input_size>=8192&&p.out_stride<65536?2u:1u;};
    uint64_t partialBytes=0,counterBytes=0;uint32_t count=0;
    for(const auto&d:command.dispatches){VulkanLinearParams p{};if(!parameters(d,p))continue;
      {
        std::lock_guard lock(impl_->workspaceMutex);
        for(uint32_t slot=1;slot<=3;++slot){const auto*w=get(d,slot);if(!w||!w->impl_)throw MetalBackendError("native single weight plane missing");const auto begin=w->gpuAddress(),end=begin+w->sizeBytes();
          for(const auto&[key,entry]:impl_->packedCaches)if(entry->source.lock()==w->impl_->allocation&&begin<entry->address+entry->bytes&&entry->address<end)throw MetalBackendError("native single requires canonical weights from a fresh process");
          impl_->canonicalPairViews[{begin,w->sizeBytes()}]=w->impl_->allocation;
        }
      }
      if(p.rows==8&&p.active_rows==8&&!p.row_start&&!p.input_f32){
        partialBytes=std::max(partialBytes,uint64_t(8)*p.out_stride*splitsFor(p)*4);counterBytes=std::max(counterBytes,uint64_t(p.out_stride/64)*4);++count;
      }
    }
    if(count){
      MetalBuffer partials,counters;
      {
        std::lock_guard lock(impl_->workspaceMutex);
        if(impl_->nativeSinglePartials.sizeBytes()<partialBytes)impl_->nativeSinglePartials=allocateBuffer(partialBytes,BufferStorage::Private,"native single partials");
        if(impl_->nativeSingleCounters.sizeBytes()<counterBytes){impl_->nativeSingleCounters=allocateBuffer(counterBytes,BufferStorage::Shared,"native single counters");std::memset(impl_->nativeSingleCounters.contents(),0,counterBytes);}
        partials=impl_->nativeSinglePartials;counters=impl_->nativeSingleCounters;
      }
      std::deque<std::array<uint32_t,52>> roots;std::vector<ComputeDispatch> expanded;expanded.reserve(command.dispatches.size());
      for(const auto&original:command.dispatches){VulkanLinearParams p{};
        if(!parameters(original,p)||p.rows!=8||p.active_rows!=8||p.row_start||p.input_f32){expanded.push_back(original);continue;}
        const uint32_t splits=splitsFor(p);MetalBuffer nativeParams;
        {
          std::lock_guard lock(impl_->workspaceMutex);const std::array<uint32_t,3> key={p.input_size,p.out_stride,splits};auto at=impl_->nativeSingleParams.find(key);
          if(at==impl_->nativeSingleParams.end()){auto b=allocateBuffer(20,BufferStorage::Shared,"native single runtime parameters");const uint32_t words[5]={p.input_size,splits,p.out_stride,0,1};std::memcpy(b.contents(),words,20);at=impl_->nativeSingleParams.emplace(key,std::move(b)).first;}nativeParams=at->second;
        }
        const uint32_t index=(p.fmt[0]-3)*4+(p.epilogue?p.epilogue+1:p.output_f32?1:0);const auto&data=kNativeSingles[index];
        auto d=original;d.pipelineName=std::string("vulkan_gguf_single_g17g_")+data.key;d.buffers.push_back({15,partials});d.buffers.push_back({16,counters});d.buffers.push_back({20,nativeParams});
        roots.emplace_back();auto&root=roots.back();std::vector<uint64_t> pointers={get(d,0)->gpuAddress(),get(d,1)->gpuAddress(),get(d,2)->gpuAddress(),get(d,3)->gpuAddress(),get(d,10)->gpuAddress(),partials.gpuAddress(),counters.gpuAddress()};
        if(p.epilogue){const auto*aux=get(d,11);if(!aux)throw MetalBackendError("native single auxiliary missing");pointers.push_back(aux->gpuAddress());}pointers.push_back(nativeParams.gpuAddress());
        for(uint32_t j=0;j<pointers.size();++j){root[2*j]=uint32_t(pointers[j]);root[2*j+1]=uint32_t(pointers[j]>>32);}
        const uint32_t scalars[13]={p.input_size&~255u,p.input_size*8,p.input_size*8+16,splits-1,0,p.input_size,splits,0,p.input_size/32,p.input_size/32/splits,1,0,p.out_stride};
        std::memcpy(root.data()+data.rootBase,scalars,sizeof(scalars));std::memcpy(root.data()+data.rootBase+13,data.literals,data.literalCount*4);
        std::erase_if(d.bytes,[](const auto&b){return b.index!=12;});d.bytes.push_back({22,root.data(),208});d.threadgroups={p.out_stride/64,splits,1};d.threadsPerThreadgroup={64,1,1};expanded.push_back(std::move(d));
      }
      if(std::getenv("SPLASH_VK_G17G_SINGLE_TRACE"))std::fprintf(stderr,"g17g_single_native_dispatches %u\n",count);
      return submitCommandAsync(Command{expanded,{}},std::move(completion));
    }
  }

  // Match Metal's staged decode policy: parallelize independent K ranges,
  // then apply the existing epilogue after the FP32 partial reduction.
  const auto stagedParameters=[](const ComputeDispatch &d,VulkanLinearParams &p){
    if(d.pipelineName!="vulkan_gguf_staged16"||d.threadgroups.z!=1||d.indirectThreadgroups.buffer)return false;
    for(const auto &b:d.buffers)if(b.index==15)return false;
    for(const auto &b:d.bytes)if(b.index==12&&b.data&&b.sizeBytes==sizeof(p)){
      std::memcpy(&p,b.data,sizeof(p));
      return vulkanRegisterSplits(p.input_size,p.out_stride)>1;
    }
    return false;
  };
  uint64_t stagedPartialBytes=0;
  for(const auto &d:command.dispatches){VulkanLinearParams p{};if(stagedParameters(d,p)){
    const uint64_t bytes=uint64_t(p.rows)*p.out_stride*vulkanRegisterSplits(p.input_size,p.out_stride)*4;
    if(bytes>UINT32_MAX)throw MetalBackendError("staged projection partial offsets exceed 32 bits");
    stagedPartialBytes=std::max(stagedPartialBytes,bytes);
  }}
  if(stagedPartialBytes){
    if(commandInFlight())throw MetalBackendError("a Vulkan command is already in flight");
    MetalBuffer partials;
    {
      std::lock_guard lock(impl_->workspaceMutex);
      if(impl_->registerPartials.sizeBytes()<stagedPartialBytes)
        impl_->registerPartials=allocateBuffer(stagedPartialBytes,BufferStorage::Private,"staged partial sums");
      partials=impl_->registerPartials;
    }
    std::vector<ComputeDispatch> expanded;expanded.reserve(command.dispatches.size()*2);
    for(const auto &original:command.dispatches){VulkanLinearParams p{};
      if(!stagedParameters(original,p)){expanded.push_back(original);continue;}
      auto d=original;d.threadgroups.z=vulkanRegisterSplits(p.input_size,p.out_stride);d.buffers.push_back({15,partials});expanded.push_back(std::move(d));
      auto reduce=original;reduce.pipelineName="vulkan_register_reduce";reduce.buffers.push_back({15,partials});reduce.threadgroups={(uint64_t(p.rows)*p.out_stride+255)/256,1,1};reduce.threadsPerThreadgroup={256,1,1};expanded.push_back(std::move(reduce));
    }
    return submitCommandAsync(Command{expanded,{}},std::move(completion));
  }
  if (const char *mode=std::getenv("SPLASH_VK_REGISTER"); mode && (std::strcmp(mode,"1")==0||std::strcmp(mode,"16")==0) &&
      device->capabilities.nativeMixedMatrix16 && device->capabilities.nativeMatrixDirectLayout) {
    const bool tensor=std::strcmp(mode,"16")==0;
    const uint32_t subgroups=vulkanRegisterSubgroups();
    const bool fusedReduction=!tensor&&std::getenv("SPLASH_VK_FUSED_REDUCE")&&std::strcmp(std::getenv("SPLASH_VK_FUSED_REDUCE"),"1")==0;
    const uint32_t rowTile=tensor?16:8;
    const bool qualified=tensor?device->capabilities.nativeBfloatMatrix16&&device->capabilities.nativeMixedMatrix16Direct:device->capabilities.nativeBfloatMatrix8;
    const auto parameters=[tensor,qualified](const ComputeDispatch &d,VulkanLinearParams &p) {
      if(!qualified||d.pipelineName!="vulkan_gguf_coop" || d.threadsPerThreadgroup.x!=128 || d.indirectThreadgroups.buffer)return false;
      for(const auto &bytes:d.bytes)if(bytes.index==12&&bytes.sizeBytes==sizeof(p)&&bytes.data){
        std::memcpy(&p,bytes.data,sizeof(p));
        if(p.input_f32||!p.input_size||p.input_size%64||!p.rows||uint64_t(p.rows)*p.out_stride*vulkanRegisterSplits(p.input_size,p.out_stride)*4>UINT32_MAX)return false;
        const uint32_t f=p.fmt[0];
        const bool supported=f==GGUF_FMT_Q6K||f==GGUF_FMT_Q4K||f==GGUF_FMT_Q5K||f==GGUF_FMT_Q80||f==GGUF_FMT_IQ4XS||f==GGUF_FMT_IQ4NL||(f>=GGUF_FMT_AF4G32&&f<=GGUF_FMT_AF4G128);
        if(!supported)return false;
        for(uint32_t i=0;i<3;++i)if(p.cols[i]&&p.fmt[i]!=f)return false;
        return true;
      }
      return false;
    };
    uint64_t tableBytes=0,sumBytes=0,partialBytes=0,counterBytes=0;
    for(const auto &d:command.dispatches){VulkanLinearParams p{};if(parameters(d,p)){
      const uint64_t tiles=(uint64_t(p.rows)+rowTile-1)/rowTile;
      tableBytes=std::max(tableBytes,tiles*rowTile*p.input_size*2);
      sumBytes=std::max(sumBytes,tiles*(p.input_size/16*rowTile+p.input_size/32*rowTile)*4);
      if(vulkanRegisterSplits(p.input_size,p.out_stride)>1){
        partialBytes=std::max(partialBytes,uint64_t(p.rows)*p.out_stride*4*vulkanRegisterSplits(p.input_size,p.out_stride));
        if(fusedReduction){uint64_t columns=0;for(auto n:p.cols)columns+=(uint64_t(n)+16*subgroups-1)/(16*subgroups);counterBytes=std::max(counterBytes,columns*tiles*4);}
      }
    }}
    if(tableBytes){
      if(commandInFlight())throw MetalBackendError("a Vulkan command is already in flight");
      MetalBuffer table,sums,partials,counters;
      {
        std::lock_guard lock(impl_->workspaceMutex);
        const auto ensure=[&](MetalBuffer &buffer,uint64_t bytes,const char *label){
          if(buffer.sizeBytes()<bytes)buffer=allocateBuffer(bytes,BufferStorage::Private,label);
        };
        ensure(impl_->registerTable,tableBytes,"register activation table");
        ensure(impl_->registerSums,sumBytes,"register activation sums");
        ensure(impl_->registerPartials,std::max(uint64_t{4},partialBytes),"register partial sums");
        if(counterBytes&&impl_->registerCounters.sizeBytes()<counterBytes){
          impl_->registerCounters=allocateBuffer(counterBytes,BufferStorage::Shared,"register reduction counters");
          std::memset(impl_->registerCounters.contents(),0,counterBytes);
        }
        table=impl_->registerTable;sums=impl_->registerSums;partials=impl_->registerPartials;
        counters=impl_->registerCounters;
      }
      std::vector<ComputeDispatch> expanded;expanded.reserve(command.dispatches.size()*5);
      std::deque<std::array<uint32_t,2>> packParams;
      std::vector<uint64_t> scheduledPacks;
      const bool cacheRequested=!tensor&&std::getenv("SPLASH_VK_Q6_NATIVE_LAYOUT")&&std::strcmp(std::getenv("SPLASH_VK_Q6_NATIVE_LAYOUT"),"1")==0;
      const bool q5Requested=!tensor&&std::getenv("SPLASH_VK_Q5_NATIVE_LAYOUT")&&std::strcmp(std::getenv("SPLASH_VK_Q5_NATIVE_LAYOUT"),"1")==0;
      const uint32_t cacheMinColumns=std::getenv("SPLASH_VK_Q6_LAYOUT_MIN_COLUMNS")?std::stoul(std::getenv("SPLASH_VK_Q6_LAYOUT_MIN_COLUMNS")):0;

      uint64_t cached=0;uint32_t cachedRows=0,cachedWidth=0,cachedStart=0,cachedActive=0;
      const bool fuseNorm=!tensor&&std::getenv("SPLASH_VK_FUSE_NORM_TABLE")&&std::strcmp(std::getenv("SPLASH_VK_FUSE_NORM_TABLE"),"1")==0;
      for(size_t at=0;at<command.dispatches.size();++at){const auto &original=command.dispatches[at];VulkanLinearParams p{};
        if(fuseNorm&&at+1<command.dispatches.size()&&
           (original.pipelineName=="norm_rms"||original.pipelineName=="norm_rms_f32")&&
           original.threadsPerThreadgroup.x==256&&original.threadsPerThreadgroup.y==1&&original.threadsPerThreadgroup.z==1&&!original.indirectThreadgroups.buffer&&
           parameters(command.dispatches[at+1],p)&&p.active_rows==8&&p.row_start==0&&p.rows%8==0&&
           original.threadgroups.x==p.rows&&original.threadgroups.y==1&&original.threadgroups.z==1){
          const auto output=std::find_if(original.buffers.begin(),original.buffers.end(),[](const auto &b){return b.index==2;});
          const auto input=std::find_if(command.dispatches[at+1].buffers.begin(),command.dispatches[at+1].buffers.end(),[](const auto &b){return b.index==0;});
          const auto width=std::find_if(original.bytes.begin(),original.bytes.end(),[](const auto &b){return b.index==3&&b.sizeBytes==4&&b.data;});
          uint32_t normWidth=0;if(width!=original.bytes.end())std::memcpy(&normWidth,width->data,4);
          if(output!=original.buffers.end()&&input!=command.dispatches[at+1].buffers.end()&&
             output->buffer.gpuAddress()==input->buffer.gpuAddress()&&normWidth==p.input_size){
            auto norm=original;norm.pipelineName=original.pipelineName=="norm_rms_f32"?"norm_rms_table16_decode_f32":"norm_rms_table16_decode";
            for(auto &b:norm.bytes)if(b.index==3)b.index=5;
            norm.buffers.push_back({3,table});norm.buffers.push_back({4,sums});expanded.push_back(std::move(norm));
            cached=output->buffer.gpuAddress();cachedRows=p.rows;cachedWidth=p.input_size;cachedStart=0;cachedActive=8;
            continue;
          }
        }
        if(!parameters(original,p)){cached=0;expanded.push_back(original);continue;}
        auto input=std::find_if(original.buffers.begin(),original.buffers.end(),[](const auto &b){return b.index==0;});
        if(input==original.buffers.end())throw MetalBackendError("register projection input is missing");
        const auto address=input->buffer.gpuAddress();
        if(address!=cached||p.rows!=cachedRows||p.input_size!=cachedWidth||p.row_start!=cachedStart||p.active_rows!=cachedActive){
          expanded.push_back({tensor?"vulkan_prepare_table16":"vulkan_prepare_table",{{0,input->buffer},{13,table},{14,sums}},original.bytes,
              {p.input_size/(tensor?32:64),(uint64_t(p.rows)+rowTile-1)/rowTile,1},{256,1,1}});
          cached=address;cachedRows=p.rows;cachedWidth=p.input_size;cachedStart=p.row_start;cachedActive=p.active_rows;
        }
        auto d=original;d.pipelineName=tensor?"vulkan_gguf_register16":"vulkan_gguf_register";d.buffers.push_back({13,table});d.buffers.push_back({14,sums});d.buffers.push_back({15,partials});
        if(((cacheRequested&&p.fmt[0]==GGUF_FMT_Q6K)||(q5Requested&&p.fmt[0]==GGUF_FMT_Q5K))&&p.plane_input_size%256==0){
          const bool q5=p.fmt[0]==GGUF_FMT_Q5K;
          const uint32_t upperBytes=q5?4:8,metaBytes=q5?16:20,unitBytes=q5?176:212;
          bool eligible=true;for(uint32_t seg=0;seg<3;++seg)if(p.cols[seg]&&p.cols[seg]<cacheMinColumns)eligible=false;
          for(uint32_t seg=0;seg<3&&eligible;++seg)if(p.cols[seg]){
            std::array<const MetalBuffer*,3> source{};
            for(uint32_t j=0;j<3;++j){auto at=std::find_if(d.buffers.begin(),d.buffers.end(),[&](const auto &b){return b.index==1+seg*3+j;});
              if(at==d.buffers.end()){eligible=false;break;}source[j]=&at->buffer;
            }
            if(!eligible)break;
            {std::lock_guard lock(impl_->workspaceMutex);auto at=impl_->canonicalPairViews.find({source[0]->gpuAddress(),source[0]->sizeBytes()});
             if(at!=impl_->canonicalPairViews.end()&&at->second.lock()==source[0]->impl_->allocation){eligible=false;break;}}
            eligible=source[0]->impl_->allocation==source[1]->impl_->allocation&&source[0]->impl_->allocation==source[2]->impl_->allocation&&
              source[1]->impl_->offset==source[0]->impl_->offset+source[0]->sizeBytes()&&
              source[2]->impl_->offset==source[1]->impl_->offset+source[1]->sizeBytes();
            const uint64_t lowStride=uint64_t(p.plane_input_size)/32*16;
            const uint64_t storedRows=source[0]->sizeBytes()/lowStride;
            eligible=eligible&&source[0]->sizeBytes()%lowStride==0&&storedRows%256==0&&storedRows>=p.cols[seg]&&
              source[1]->sizeBytes()==storedRows*p.plane_input_size/32*upperBytes&&source[2]->sizeBytes()==storedRows*p.plane_input_size/256*metaBytes&&
              storedRows*p.plane_input_size/256*unitBytes<=UINT32_MAX;
          }
          if(eligible){
            auto packedDispatch=d;std::vector<ComputeDispatch> newPacks;
            for(uint32_t seg=0;seg<3&&eligible;++seg)if(p.cols[seg]){
              std::array<MetalBuffer,3> src;for(uint32_t j=0;j<3;++j){
                auto b=std::find_if(d.buffers.begin(),d.buffers.end(),[&](const auto &v){return v.index==1+seg*3+j;});
                if(b==d.buffers.end())throw MetalBackendError("packed cache source missing");src[j]=b->buffer;
              }
              const uint64_t storedRows=src[0].sizeBytes()/(uint64_t(p.plane_input_size)/32*16);
              const uint64_t size=storedRows*p.plane_input_size/256*unitBytes;
              if(storedRows%256||storedRows<p.cols[seg]||size>UINT32_MAX){eligible=false;break;}
              const std::array<uint64_t,8> key={src[0].gpuAddress(),src[1].gpuAddress(),src[2].gpuAddress(),src[0].sizeBytes(),src[1].sizeBytes(),src[2].sizeBytes(),p.plane_input_size,uint64_t(tensor)|(q5?2u:0u)};
              std::shared_ptr<PackedCacheEntry> entry;
              {
                std::lock_guard lock(impl_->workspaceMutex);
                for(auto at=impl_->packedCaches.begin();at!=impl_->packedCaches.end();){
                  bool dead=false;dead=at->second->source.expired();
                  if(dead)at=impl_->packedCaches.erase(at);else ++at;
                }
                if(auto found=impl_->packedCaches.find(key);found!=impl_->packedCaches.end())entry=found->second;
                else{
                  for(const auto &[previousKey,previous]:impl_->packedCaches){
                    if(previous->source.lock()!=src[0].impl_->allocation)continue;
                    const auto begin=src[0].gpuAddress();
                    if(begin<previous->address+previous->bytes&&previous->address<begin+size)
                      throw MetalBackendError("in-place quantized does not permit changing a transformed weight view or matrix layout");
                  }
                  entry=std::make_shared<PackedCacheEntry>();entry->address=src[0].gpuAddress();entry->bytes=size;
                  entry->source=src[0].impl_->allocation;
                  impl_->packedCaches.emplace(key,entry);
                }
              }
              const auto address=entry->address;
              if(src[0].impl_->allocation!=src[1].impl_->allocation||src[0].impl_->allocation!=src[2].impl_->allocation||
                 src[1].impl_->offset!=src[0].impl_->offset+src[0].sizeBytes()||src[2].impl_->offset!=src[1].impl_->offset+src[1].sizeBytes()||
                 src[0].sizeBytes()+src[1].sizeBytes()+src[2].sizeBytes()!=size)
                throw MetalBackendError("in-place quantized needs contiguous, exclusively used canonical planes");
              auto viewImpl=std::make_shared<MetalBuffer::Impl>();viewImpl->allocation=src[0].impl_->allocation;
              viewImpl->offset=src[0].impl_->offset;viewImpl->bytes=size;MetalBuffer packedView(viewImpl);
              if(!entry->ready&&std::find(scheduledPacks.begin(),scheduledPacks.end(),address)==scheduledPacks.end()){
                MetalBuffer scratch;
                {std::lock_guard lock(impl_->workspaceMutex);
                 if(impl_->registerCacheScratch.sizeBytes()<size)impl_->registerCacheScratch=allocateBuffer(size,BufferStorage::Private,"in-place quantized repack scratch");
                 scratch=impl_->registerCacheScratch;}
                packParams.push_back({p.plane_input_size,uint32_t(storedRows)});
                newPacks.push_back({q5?"vulkan_q5_repack8":tensor?"vulkan_q6_repack16":"vulkan_q6_repack8",{{0,src[0]},{1,src[1]},{2,src[2]},{3,scratch}},{{4,packParams.back().data(),8}},
                  {std::min(uint64_t{65535},(storedRows*p.plane_input_size/256+3)/4),((storedRows*p.plane_input_size/256+3)/4+65534)/65535,1},{128,1,1}});
                packParams.push_back({uint32_t(size/16),0});uint64_t groups=(size/16+255)/256;
                newPacks.push_back({"vulkan_q6_repack_commit",{{0,scratch},{1,packedView}},{{2,packParams.back().data(),4}},
                  {std::min(uint64_t{65535},groups),(groups+65534)/65535,1},{256,1,1}});
                scheduledPacks.push_back(address);
              }
              for(auto &b:packedDispatch.buffers)if(b.index==1+seg*3)b.buffer=packedView;
            }
            if(eligible){
              for(auto &pack:newPacks)expanded.push_back(std::move(pack));
              packedDispatch.buffers.push_back({17,table});d=std::move(packedDispatch);
            }else{
              // Packs abandoned by an incomplete segment group must be retried.
              for(const auto &pack:newPacks)if(pack.pipelineName=="vulkan_q6_repack_commit")for(const auto &b:pack.buffers)if(b.index==1)
                scheduledPacks.erase(std::remove(scheduledPacks.begin(),scheduledPacks.end(),b.buffer.gpuAddress()),scheduledPacks.end());
            }
          }
        }
        const uint32_t columnsPerGroup=16*subgroups;
        uint64_t tiles=0;for(auto cols:p.cols)tiles+=(uint64_t(cols)+columnsPerGroup-1)/columnsPerGroup;d.threadgroups.x=tiles;d.threadgroups.y=(uint64_t(p.rows)+rowTile-1)/rowTile;d.threadgroups.z=vulkanRegisterSplits(p.input_size,p.out_stride);d.threadsPerThreadgroup={32*subgroups,1,1};
        if(fusedReduction&&d.threadgroups.z>1)d.buffers.push_back({16,counters});
        expanded.push_back(std::move(d));
        if(!fusedReduction&&vulkanRegisterSplits(p.input_size,p.out_stride)>1){auto reduce=original;reduce.pipelineName="vulkan_register_reduce";reduce.buffers.push_back({15,partials});reduce.threadgroups={(uint64_t(p.rows)*p.out_stride+255)/256,1,1};reduce.threadsPerThreadgroup={256,1,1};expanded.push_back(std::move(reduce));}
        for(const auto &binding:original.buffers)if(binding.index==10){
          const auto begin=binding.buffer.gpuAddress(),end=begin+binding.buffer.sizeBytes();
          const auto inputEnd=cached+(uint64_t(p.row_start)+p.rows)*p.input_size*2;
          if(begin<inputEnd&&cached<end)cached=0;
        }
      }
      return submitCommandAsync(Command{expanded,{}},std::move(completion));
    }
  }
  if (const char *mode=std::getenv("SPLASH_VK_F32_INPUT"); mode && std::strcmp(mode,"1")==0) {
    uint64_t maximum=0;
    const auto eligible=[](const ComputeDispatch &d) {
      if(d.pipelineName!="vulkan_gguf_linear" || d.threadsPerThreadgroup.x!=256) return false;
      for(const auto &bytes:d.bytes)if(bytes.index==12&&bytes.sizeBytes==sizeof(VulkanLinearParams)) {
        VulkanLinearParams p;std::memcpy(&p,bytes.data,sizeof(p));return !p.input_f32;
      }
      return false;
    };
    for(const auto &d:command.dispatches)if(eligible(d))for(const auto &b:d.buffers)if(b.index==0)maximum=std::max(maximum,b.buffer.sizeBytes()*2);
    if(maximum) {
      if(commandInFlight()) throw MetalBackendError("a Vulkan command is already in flight");
      auto scratch=allocateBuffer(maximum,BufferStorage::Private,"expanded Vulkan input");
      std::deque<VulkanLinearParams> parameters;
      std::deque<uint32_t> counts;
      std::vector<ComputeDispatch> expanded;
      expanded.reserve(command.dispatches.size()*2);
      uint64_t cachedAddress=0,cachedBytes=0;
      for(const auto &original:command.dispatches) {
        auto d=original;
        if(!eligible(d)){cachedAddress=0;cachedBytes=0;expanded.push_back(std::move(d));continue;}
        auto input=std::find_if(d.buffers.begin(),d.buffers.end(),[](const auto &b){return b.index==0;});
        if(input==d.buffers.end()||!input->buffer.gpuAddress()||input->buffer.sizeBytes()%2||input->buffer.sizeBytes()/2>UINT32_MAX)
          throw MetalBackendError("invalid expandable projection input");
        const auto address=input->buffer.gpuAddress(),bytes=input->buffer.sizeBytes();
        if(address!=cachedAddress||bytes!=cachedBytes){
          counts.push_back(uint32_t(bytes/2));
          expanded.push_back({"vulkan_widen_input",{{0,input->buffer},{1,scratch}},{{2,&counts.back(),4}},
                              {(uint64_t(counts.back())+255)/256,1,1},{256,1,1}});
          cachedAddress=address;cachedBytes=bytes;
        }
        input->buffer=scratch;
        auto params=std::find_if(d.bytes.begin(),d.bytes.end(),[](const auto &b){return b.index==12;});
        parameters.emplace_back();std::memcpy(&parameters.back(),params->data,sizeof(VulkanLinearParams));parameters.back().input_f32=1;
        params->data=&parameters.back();
        for(const auto &b:d.buffers)if(b.index==10){const auto begin=b.buffer.gpuAddress(),end=begin+b.buffer.sizeBytes();
          if(begin<cachedAddress+cachedBytes&&cachedAddress<end){cachedAddress=0;cachedBytes=0;}}
        expanded.push_back(std::move(d));
      }
      return submitCommandAsync(Command{expanded,{}},std::move(completion));
    }
  }
  if (device->inFlight.exchange(true)) throw MetalBackendError("a Vulkan command is already in flight");
  auto state = std::make_shared<CommandTicket::State>();
  state->owner = device;
  state->generation = ++device->generation;
  state->start = Clock::now();
  if (const char *path = std::getenv("SPLASH_VK_PROFILE"); path && *path) {
    state->profilePath = path;
    for (const auto &dispatch : command.dispatches) {
      std::string name = dispatch.pipelineName;
      if (name == "vulkan_gguf_linear" || name == "vulkan_gguf_coop" || name == "vulkan_gguf_register" || name == "vulkan_gguf_register16" || name == "vulkan_gguf_staged16" || name == "vulkan_gguf_direct16") for (const auto &bytes : dispatch.bytes)
        if (bytes.index == 12 && bytes.sizeBytes == sizeof(VulkanLinearParams)) {
          VulkanLinearParams p; std::memcpy(&p, bytes.data, sizeof(p));
          name += ":rows=" + std::to_string(p.rows) + ":k=" + std::to_string(p.input_size) +
                  ":n=" + std::to_string(p.out_stride) + ":f=" + std::to_string(p.fmt[0]) +
                  ":ep=" + std::to_string(p.epilogue);
        }
      state->profileNames.push_back(std::move(name));
    }
  }
  try {
    {
      std::lock_guard lock(impl_->workspaceMutex);
      std::erase_if(impl_->packedCaches,[](const auto &item){return item.second->source.expired();});
    }
    std::vector<VkPipeline> preparedPipelines;
    std::vector<VkShaderEXT> preparedNative;
    std::vector<uint64_t> nativeRoots;
    preparedPipelines.reserve(command.dispatches.size());
    preparedNative.reserve(command.dispatches.size());nativeRoots.reserve(command.dispatches.size());
    constexpr uint32_t bindings = 32;
    std::vector<uint32_t> indices;
    uint64_t arenaBytes = 0;
    for (const auto &dispatch : command.dispatches) {
      arenaBytes = aligned(arenaBytes, 16);
      if (arenaBytes / 8 > UINT32_MAX) throw MetalBackendError("Vulkan dispatch binding arena is too large");
      indices.push_back(uint32_t(arenaBytes / 8));
      arenaBytes += bindings * sizeof(uint64_t);
      for (const auto &bytes : dispatch.bytes) arenaBytes = aligned(arenaBytes, 16) + bytes.sizeBytes;
    }
    auto arena = allocateBuffer(std::max(uint64_t{16}, arenaBytes), BufferStorage::Shared, "dispatch bindings");
    std::memset(arena.contents(), 0, arena.sizeBytes());
    for (size_t i = 0; i < command.dispatches.size(); ++i) {
      const auto &dispatch = command.dispatches[i];
      const bool packedProjection=(dispatch.pipelineName=="vulkan_gguf_register"||dispatch.pipelineName=="vulkan_gguf_register16")&&
        std::any_of(dispatch.buffers.begin(),dispatch.buffers.end(),[](const auto &b){return b.index==17;});
      if(!impl_->packedCaches.empty()){
        std::lock_guard lock(impl_->workspaceMutex);
        const auto first=impl_->packedCaches.begin()->second->address;
        const auto &last=impl_->packedCaches.rbegin()->second;
        const auto limit=last->address+last->bytes;
        for(const auto &binding:dispatch.buffers){
          if(!binding.buffer.impl_||binding.buffer.sizeBytes()==0)continue;
          if((packedProjection||dispatch.pipelineName=="vulkan_register_reduce")&&binding.index>=1&&binding.index<=9)continue;
          const auto begin=binding.buffer.gpuAddress(),end=begin+binding.buffer.sizeBytes();
          if(end<=first||begin>=limit)continue;
          std::array<uint64_t,8> probe{};probe[0]=begin;
          auto at=impl_->packedCaches.lower_bound(probe);
          if(at!=impl_->packedCaches.begin())--at;
          for(;at!=impl_->packedCaches.end()&&at->first[0]<end;++at){
            const auto &entry=at->second;
            if(entry->address+entry->bytes<=begin||entry->source.lock()!=binding.buffer.impl_->allocation)continue;
            const bool initializing=!entry->ready&&binding.index<=2&&
              (dispatch.pipelineName=="vulkan_q6_repack8"||dispatch.pipelineName=="vulkan_q6_repack16"||dispatch.pipelineName=="vulkan_q5_repack8");
            const bool committing=!entry->ready&&dispatch.pipelineName=="vulkan_q6_repack_commit"&&binding.index==1&&
              begin==entry->address&&binding.buffer.sizeBytes()==entry->bytes;
            if(initializing||committing)continue;
            throw MetalBackendError("kernel cannot access a transformed immutable quantized weight view: "+dispatch.pipelineName);
          }
        }
      }
      auto *addresses = static_cast<uint64_t *>(arena.contents()) + indices[i];
      std::array<bool, bindings> used{};
      for (const auto &binding : dispatch.buffers) {
        if (binding.index >= bindings || used[binding.index] || !binding.buffer.impl_ ||
            binding.buffer.impl_->allocation->owner != device || !binding.buffer.gpuAddress())
          throw MetalBackendError("invalid Vulkan buffer binding");
        used[binding.index] = true;
        addresses[binding.index] = binding.buffer.gpuAddress();
        state->resources.push_back(binding.buffer);
        if(binding.index==16&&dispatch.pipelineName=="vulkan_gguf_register"&&
           std::getenv("SPLASH_VK_CHECK_COUNTERS")&&std::strcmp(std::getenv("SPLASH_VK_CHECK_COUNTERS"),"1")==0)
          state->checkedCounters=binding.buffer;
      }
      uint64_t offset = uint64_t(indices[i]) * 8 + bindings * sizeof(uint64_t);
      for (const auto &bytes : dispatch.bytes) {
        if (bytes.index >= bindings || used[bytes.index] || !bytes.data || !bytes.sizeBytes)
          throw MetalBackendError("invalid Vulkan constant binding");
        used[bytes.index] = true;
        offset = aligned(offset, 16);
        std::memcpy(static_cast<std::byte *>(arena.contents()) + offset, bytes.data, bytes.sizeBytes);
        addresses[bytes.index] = arena.gpuAddress() + offset;
        offset += bytes.sizeBytes;
      }
      auto native=device->nativePair(dispatch);preparedNative.push_back(native);
      nativeRoots.push_back(native?addresses[22]:0);
      if(native&&!nativeRoots.back())throw MetalBackendError("native Root arena binding missing");
      preparedPipelines.push_back(native?VK_NULL_HANDLE:device->pipeline(dispatch));
      if(dispatch.pipelineName=="vulkan_q6_repack_commit"){
        for(const auto &b:dispatch.buffers)if(b.index==1){std::lock_guard lock(impl_->workspaceMutex);
          for(const auto &[key,entry]:impl_->packedCaches)if(entry->address==b.buffer.gpuAddress())state->preparedCaches.push_back(entry);
        }
      }

    }
    state->resources.push_back(arena);
    {
      std::lock_guard lock(device->mutex);
      for (const auto &entry : device->allocations)
        if (auto allocation = entry.lock()) state->resident.push_back(std::move(allocation));
    }
    VkCommandPoolCreateInfo pool{};
    pool.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool.queueFamilyIndex = device->queueFamily;
    pool.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    checked(vkCreateCommandPool(device->device, &pool, nullptr, &state->pool), "create command pool");
    VkCommandBufferAllocateInfo allocation{};
    allocation.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocation.commandPool = state->pool;
    allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocation.commandBufferCount = 1;
    // Bound each kernel-scheduler job. A complete 2048-token model prefill
    // is a single logical Splash command but exceeds Linux's compute-job
    // timeout when encoded into one Vulkan submission.
    const bool debugSync=std::getenv("SPLASH_VK_DEBUG_SYNC") && std::strcmp(std::getenv("SPLASH_VK_DEBUG_SYNC"),"1")==0;
    size_t matrixDispatchLimit = 4;
    if (const char *value = std::getenv("SPLASH_VK_MATRIX_DISPATCHES")) {
      char *end = nullptr;
      const unsigned long parsed = std::strtoul(value, &end, 10);
      if (!value[0] || *end || parsed < 1 || parsed > 64)
        throw MetalBackendError("SPLASH_VK_MATRIX_DISPATCHES must be 1 through 64");
      matrixDispatchLimit = parsed;
    }
    size_t dispatchesPerBatch = 64;
    for (const auto &dispatch : command.dispatches) {
      if (dispatch.pipelineName.starts_with("vulkan_gguf_pair_") || dispatch.pipelineName.starts_with("vulkan_gguf_single_g17g_") || dispatch.pipelineName == "vulkan_gguf_coop" || dispatch.pipelineName == "vulkan_gguf_register" || dispatch.pipelineName == "vulkan_gguf_register16" || dispatch.pipelineName == "vulkan_gguf_staged16" || dispatch.pipelineName == "vulkan_gguf_direct16")
        dispatchesPerBatch = std::min(dispatchesPerBatch, matrixDispatchLimit);
      if (dispatch.pipelineName != "vulkan_gguf_linear") continue;
      for (const auto &bytes : dispatch.bytes) if (bytes.index == 12 && bytes.sizeBytes == sizeof(VulkanLinearParams)) {
        VulkanLinearParams p; std::memcpy(&p, bytes.data, sizeof(p));
        if (p.rows >= 64 || p.row_start) dispatchesPerBatch = 4;
      }
    }
    if(debugSync)dispatchesPerBatch=1;
    std::vector<VkCommandBuffer> batches;
    VkCommandBuffer buffer;
    checked(vkAllocateCommandBuffers(device->device, &allocation, &buffer), "allocate command buffer");
    VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
    VkDescriptorPoolCreateInfo descriptorPool{};
    descriptorPool.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPool.maxSets = 1;
    descriptorPool.poolSizeCount = 1;
    descriptorPool.pPoolSizes = &poolSize;
    checked(vkCreateDescriptorPool(device->device, &descriptorPool, nullptr, &state->descriptors), "create descriptor pool");
    VkDescriptorSetAllocateInfo descriptor{};
    descriptor.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptor.descriptorPool = state->descriptors;
    descriptor.descriptorSetCount = 1;
    descriptor.pSetLayouts = &device->descriptorLayout;
    VkDescriptorSet set;
    checked(vkAllocateDescriptorSets(device->device, &descriptor, &set), "allocate descriptor set");
    VkDescriptorBufferInfo descriptorBuffer{arena.impl_->allocation->buffer, 0, arena.sizeBytes()};
    VkWriteDescriptorSet write{};
    write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    write.dstSet = set;
    write.dstBinding = 0;
    write.descriptorCount = 1;
    write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo = &descriptorBuffer;
    vkUpdateDescriptorSets(device->device, 1, &write, 0, nullptr);
    VkQueryPoolCreateInfo query{};
    query.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    query.queryType = VK_QUERY_TYPE_TIMESTAMP;
    query.queryCount = uint32_t(2 + 2 * state->profileNames.size());
    checked(vkCreateQueryPool(device->device, &query, nullptr, &state->queries), "create timestamp pool");
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    checked(vkBeginCommandBuffer(buffer, &begin), "begin command");
    vkCmdResetQueryPool(buffer, state->queries, 0, query.queryCount);
    vkCmdWriteTimestamp(buffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, state->queries, 0);
    vkCmdBindDescriptorSets(buffer, VK_PIPELINE_BIND_POINT_COMPUTE, device->pipelineLayout, 0, 1, &set, 0, nullptr);
    for (size_t i = 0; i < command.dispatches.size(); ++i) {
      if (i && i % dispatchesPerBatch == 0) {
        checked(vkEndCommandBuffer(buffer), "end compute batch");
        batches.push_back(buffer);
        checked(vkAllocateCommandBuffers(device->device, &allocation, &buffer), "allocate compute batch");
        checked(vkBeginCommandBuffer(buffer, &begin), "begin compute batch");
        vkCmdBindDescriptorSets(buffer, VK_PIPELINE_BIND_POINT_COMPUTE, device->pipelineLayout, 0, 1, &set, 0, nullptr);
      }
      const auto &dispatch = command.dispatches[i];
      VkMemoryBarrier barrier{};
      barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
      barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT;
      vkCmdPipelineBarrier(buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
          VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, 0, 1, &barrier, 0, nullptr, 0, nullptr);
      if(preparedNative[i]){
        const VkShaderStageFlagBits stage=VK_SHADER_STAGE_COMPUTE_BIT;
        device->bindShaders(buffer,1,&stage,&preparedNative[i]);
        vkCmdPushConstants(buffer,device->nativePairLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,8,&nativeRoots[i]);
      }else{
        vkCmdBindPipeline(buffer,VK_PIPELINE_BIND_POINT_COMPUTE,preparedPipelines[i]);
        vkCmdPushConstants(buffer,device->pipelineLayout,VK_SHADER_STAGE_COMPUTE_BIT,0,sizeof(uint32_t),&indices[i]);
      }
      if (!state->profilePath.empty())
        vkCmdWriteTimestamp(buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, state->queries, uint32_t(2 + 2 * i));
      if (dispatch.indirectThreadgroups.buffer) {
        const auto &indirect = dispatch.indirectThreadgroups;
        if (!indirect.buffer.impl_ || indirect.buffer.impl_->allocation->owner != device ||
            indirect.offsetBytes > indirect.buffer.sizeBytes() ||
            indirect.buffer.sizeBytes() - indirect.offsetBytes < sizeof(VkDispatchIndirectCommand) ||
            (indirect.buffer.impl_->offset + indirect.offsetBytes) % 4)
          throw MetalBackendError("invalid indirect Vulkan grid");
        vkCmdDispatchIndirect(buffer, indirect.buffer.impl_->allocation->buffer, indirect.buffer.impl_->offset + indirect.offsetBytes);
        state->resources.push_back(indirect.buffer);
      } else {
        const auto groups = dispatch.threadgroups;
        if (groups.x > device->properties.limits.maxComputeWorkGroupCount[0] ||
            groups.y > device->properties.limits.maxComputeWorkGroupCount[1] ||
            groups.z > device->properties.limits.maxComputeWorkGroupCount[2])
          throw MetalBackendError("Vulkan dispatch grid exceeds device limits");
        vkCmdDispatch(buffer, uint32_t(groups.x), uint32_t(groups.y), uint32_t(groups.z));
      }
      if (!state->profilePath.empty())
        vkCmdWriteTimestamp(buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, state->queries, uint32_t(3 + 2 * i));
    }
    VkMemoryBarrier host{};
    host.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    host.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    host.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &host, 0, nullptr, 0, nullptr);
    vkCmdWriteTimestamp(buffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, state->queries, 1);
    checked(vkEndCommandBuffer(buffer), "end command");
    VkFenceCreateInfo fence{};
    fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    checked(vkCreateFence(device->device, &fence, nullptr, &state->fence), "create fence");
    batches.push_back(buffer);
    std::vector<VkSubmitInfo> submits(batches.size());
    for (size_t i = 0; i < batches.size(); ++i) {
      submits[i].sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
      submits[i].commandBufferCount = 1;
      submits[i].pCommandBuffers = &batches[i];
    }
    state->submitted = Clock::now();
    device->startedNs = std::chrono::duration_cast<std::chrono::nanoseconds>(state->submitted.time_since_epoch()).count();
    device->gpuDone = false;
    {
      std::lock_guard lock(device->queueMutex);
      if(debugSync){
        for(size_t i=0;i<submits.size();++i){
          const auto &d=command.dispatches[i];
          std::fprintf(stderr,"VK DEBUG command=%llu dispatch=%zu %s groups=%llu,%llu,%llu\n",(unsigned long long)state->generation,i,d.pipelineName.c_str(),(unsigned long long)d.threadgroups.x,(unsigned long long)d.threadgroups.y,(unsigned long long)d.threadgroups.z);
          checked(vkQueueSubmit(device->queue,1,&submits[i],state->fence),"debug submit");
          checked(vkWaitForFences(device->device,1,&state->fence,true,UINT64_MAX),"debug dispatch completion");
          if(i+1<submits.size())checked(vkResetFences(device->device,1,&state->fence),"reset debug fence");
        }
      }else{
      checked(vkQueueSubmit(device->queue, uint32_t(submits.size()), submits.data(), state->fence), "submit command");
      }
    }
    auto finish = [state, completion = std::move(completion)] {
      const auto status = vkWaitForFences(state->owner->device, 1, &state->fence, true, UINT64_MAX);
      std::string error;
      CommandTiming timing;
      if (status != VK_SUCCESS) error = "Vulkan command failed: " + std::to_string(status);
      else {
        uint64_t timestamps[2]{};
        const auto query = vkGetQueryPoolResults(state->owner->device, state->queries, 0, 2,
            sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
        if (query != VK_SUCCESS) error = "Vulkan timestamp readback failed";
        else {
          const auto bits = state->owner->timestampBits;
          const uint64_t mask = bits == 64 ? UINT64_MAX : (uint64_t{1} << bits) - 1;
          timing.gpuSeconds = double((timestamps[1] - timestamps[0]) & mask) * state->owner->properties.limits.timestampPeriod * 1e-9;
          if (!state->profilePath.empty()) {
            std::vector<uint64_t> values(2 * state->profileNames.size());
            const auto result = vkGetQueryPoolResults(state->owner->device, state->queries, 2, uint32_t(values.size()),
                values.size() * sizeof(uint64_t), values.data(), sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
            if (result == VK_SUCCESS) {
              std::lock_guard lock(state->owner->mutex);
              std::ofstream profile(state->profilePath, std::ios::app);
              for (size_t i = 0; i < state->profileNames.size(); ++i)
                profile << state->generation << '\t' << i << '\t' << state->profileNames[i] << '\t'
                        << double((values[2 * i + 1] - values[2 * i]) & mask) *
                               state->owner->properties.limits.timestampPeriod * 1e-3 << '\n';
            }
          }
        }
      }
      // Drop GPU lifetime holds before publishing completion. A caller may
      // replace a completed ticket only after submitting its next command;
      // retaining the old resident snapshot until destruction transfers dead
      // allocations into each succeeding ticket's snapshot.
      if(status==VK_SUCCESS){
        if(state->checkedCounters.sizeBytes()){
          const auto *values=static_cast<const uint32_t*>(state->checkedCounters.contents());
          for(uint64_t i=0;i<state->checkedCounters.sizeBytes()/4;++i)if(values[i]){error="fused reduction left a nonzero tile counter";break;}
        }
        if(error.empty())for(const auto &entry:state->preparedCaches)entry->ready=true;
        state->preparedCaches.clear();state->checkedCounters={};state->resources.clear();state->resident.clear();
      }
      timing.wallSeconds = std::chrono::duration<double>(Clock::now() - state->start).count();
      if (!error.empty()) state->owner->fail(error);
      {
        std::lock_guard lock(state->mutex);
        state->error = std::move(error);
        state->timing = timing;
        if (state->owner->generation.load() == state->generation) state->owner->gpuDone = true;
        state->done = true;
      }
      state->condition.notify_all();
      if (completion) {
        try { completion(); } catch (...) { state->owner->fail("Vulkan completion callback threw"); }
      }
    };
    std::thread worker;
    try {
      worker = std::thread(std::move(finish));
      worker.detach();
    } catch (...) {
      if (worker.joinable()) worker.join();
      else vkWaitForFences(device->device, 1, &state->fence, true, UINT64_MAX);
      device->gpuDone = true;
      throw;
    }
  } catch (...) {
    device->inFlight = false;
    throw;
  }
  return CommandTicket(std::move(state));
}
void MetalBackend::preparePipelines(std::span<const ComputeDispatch> dispatches) {
  checkOperation();
  for (const auto &dispatch : dispatches) (void)(impl_->device->nativePair(dispatch)?VK_NULL_HANDLE:impl_->device->pipeline(dispatch));
}
MetalMemoryStats MetalBackend::memoryStats() const noexcept {
  std::lock_guard lock(impl_->device->mutex);
  return impl_->device->stats;
}
MetalMemoryStats MetalBackend::refreshMemoryStats() const noexcept { return memoryStats(); }
bool MetalBackend::commandInFlight() const noexcept { return impl_->device->inFlight.load(); }
bool MetalBackend::healthy() const noexcept { return impl_->device->healthy.load(); }
void MetalBackend::checkHealth() {
  auto device = impl_->device;
  if (device->inFlight.load() && !device->gpuDone.load() &&
      std::chrono::duration<double>(Clock::now().time_since_epoch()).count() - device->startedNs.load() * 1e-9 > device->timeout)
    device->fail("Vulkan command timed out");
  if (!device->healthy.load()) throw MetalBackendError(unhealthyReason());
}
std::string MetalBackend::unhealthyReason() const {
  std::lock_guard lock(impl_->device->mutex);
  return impl_->device->reason;
}

} // namespace splash::metal
