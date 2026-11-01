#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require
layout(constant_id=3) const bool VERIFY=false;
layout(constant_id=4) const uint KH=4;
layout(constant_id=5) const uint QH=24;
layout(constant_id=6) const bool NORM_F32=false;
float rb(uint64_t a,uint i){return bf16_float(R16(a).values[i]);}
void wb(uint64_t a,uint i,float v){W16(a).values[i]=float_bf16(v);}
uint64_t page_address(uint64_t table,uint token,uint head,uint dim,uint extent_pages,uint layer_offset,bool value){
 uint64_t entry=RU64(table).values[token/32];uint index=uint(entry)&16383;
 uint64_t base=entry-index+layer_offset;
 uint pageBytes=KH*32*256*2;
 if(value)return base+uint64_t(extent_pages)*pageBytes+uint64_t(index)*pageBytes+2*((head*256+dim)*32+token%32);
 return base+uint64_t(index)*pageBytes+2*((head*32+token%32)*256+dim);
}
