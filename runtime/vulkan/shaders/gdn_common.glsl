#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require
layout(constant_id=3) const bool NORM_F32=false;
layout(constant_id=4) const uint VH=48;
layout(constant_id=5) const bool STATE_BF16=false;
const uint KW=2048;
uint vw(){return VH*128;}
uint cd(){return 4096+vw();}
uint pw(){return VH==48?16640:12544;}
float rb(uint64_t a,uint i){return bf16_float(R16(a).values[i]);}
void wb(uint64_t a,uint i,float v){W16(a).values[i]=float_bf16(v);}
float round_b(float v){return bf16_float(float_bf16(v));}
float silu(float x){return x/(1.0+exp(-x));}
float norm_weight(uint64_t a,uint i){return NORM_F32?RF32(a).values[i]:rb(a,i);}
float carried(uint64_t state,uint64_t rows,uint stride,uint consumed,uint row,uint ch){
 uint source=consumed+row;
 return source<3?rb(state,source*cd()+ch):rb(rows,(source-3)*stride+ch);
}
float conv(uint64_t packed,uint64_t weights,uint64_t state,uint64_t history,uint stride,uint consumed,uint t,uint ch){
 float v=0;
 for(uint tap=0;tap<4;++tap){uint pos=t+tap; float x=pos<3?carried(state,history,stride,consumed,pos,ch):rb(packed,(pos-3)*pw()+ch);v+=x*rb(weights,ch*4+tap);}
 return round_b(silu(round_b(v)));
}
vec2 gates(uint64_t packed,uint64_t scale,uint64_t bias,uint t,uint h){
 float b=rb(packed,t*pw()+cd()+vw()+h);
 float x=round_b(rb(packed,t*pw()+cd()+vw()+VH+h)+rb(bias,h));
 float soft=round_b(max(x,0.0)+log(1.0+exp(-abs(x))));
 return vec2(exp(RF32(scale).values[h]*soft),round_b(1.0/(1.0+exp(-b))));
}
float state_read(uint64_t a,uint i){return STATE_BF16?rb(a,i):RF32(a).values[i];}
void state_write(uint64_t a,uint i,float v){if(STATE_BF16)wb(a,i,v);else WF32(a).values[i]=v;}
