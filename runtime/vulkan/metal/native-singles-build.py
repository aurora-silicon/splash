from pathlib import Path
import subprocess,json,time,hashlib,sys
p=Path(__file__).parent;repo=Path('/home/dj/Work/splash-linux-vulkan-20261014');report=p/'build-receipt.json';assert not report.exists();a={'status':'running','target':'applegpu_g17g','steps':[],'scope':'Compile original Metal Rows8Q6/Q5 single projections with verified capacity16 and all epilogues; G17G only'}
win=lambda f:'Z:'+str(f).replace('/','\\');vm=['muvm','-i','--gpu-mode=software','--mem=2048','--','/home/dj/Work/m5-gpu-20260930/mesa-parity-20261001/local-compiler/metal-guest.sh'];rt=Path('/home/dj/Work/metal-parity-g17g-20261014/staged-probe-20261015/libTensorOpsInlined26.rtlib');framework=Path('/home/dj/Work/m5-gpu-20260930/macos-cross-sdk/MacOSX26.1.sdk/System/Library/Frameworks')
for mode in ['fixed16']:
 flags=['-std=metal4.0','-O3','-Wno-unused-parameter']
 if mode=='fixed16':flags+=['-I'+win(p/'include')]
 flags+=['-I'+win(repo/'runtime'),'-F'+win(framework),'-target','air64-apple-macosx26.4']
 functions=['gguf_decode_q6k_m8_a', 'gguf_decode_q6k_m8_a_f32', 'gguf_decode_q6k_m8_r', 'gguf_decode_q6k_m8_g', 'gguf_decode_q5k_m8_a', 'gguf_decode_q5k_m8_a_f32', 'gguf_decode_q5k_m8_r', 'gguf_decode_q5k_m8_g'];(p/f'{mode}.mtlp-json').write_text(json.dumps({'libraries':{'paths':[{'label':'L','path':str(p/f'{mode}.metallib')}]},'pipelines':{'compute_pipelines':[{'compute_function':'alias:L#'+function}for function in functions]}})+'\n')
 steps=[('metal',flags+['-c',win(p/f'{mode}.metal'),'-o',win(p/f'{mode}.air')]),('metallib',[win(p/f'{mode}.air'),win(rt),'-o',win(p/f'{mode}.metallib')]),('metal-nt',['-arch','applegpu_g17g','-platform_version','macos','26.4','26.4','-N',win(p/f'{mode}.mtlp-json'),'-o',win(p/f'{mode}.gpubin')])]
 for tool,args in steps:
  log=p/f'{mode}-{tool}.log';start=time.monotonic()
  with log.open('x')as f:r=subprocess.run(vm+[tool,*args],stdout=f,stderr=subprocess.STDOUT,timeout=300)
  a['steps'].append({'mode':mode,'tool':tool,'exit':r.returncode,'seconds':time.monotonic()-start,'command':vm+[tool,*args]});report.write_text(json.dumps(a,indent=2)+'\n');print(json.dumps(a['steps'][-1]),flush=True)
  if r.returncode:a['status']='failed';report.write_text(json.dumps(a,indent=2)+'\n');print(log.read_text()[-2000:],flush=True);sys.exit(r.returncode)
a['status']='compiled; capacity value must be verified beforeusing fixed16, then numerical gates required';report.write_text(json.dumps(a,indent=2)+'\n')
