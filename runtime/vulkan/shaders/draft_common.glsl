#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require
float rb(uint64_t a,uint i){return bf16_float(R16(a).values[i]);}
void wb(uint64_t a,uint i,float v){W16(a).values[i]=float_bf16(v);}
