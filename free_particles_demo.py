import sys, time
import numpy as np
from PIL import Image
import imageio.v2 as imageio
S='.'
sys.path.insert(0, S)
from free_particles import FreeLayout

def run(tag, rgb, steps=900, k_rep=2.0):
    fl = FreeLayout(rgb, lam=1.0, k_rep=k_rep)
    w = imageio.get_writer(f'{S}/free2_{tag}.mp4', fps=30, quality=8, macro_block_size=1)
    t0=time.time()
    for i in range(steps):
        mv = fl.step(friction=0.25)
        if i % 3 == 0: w.append_data(fl.render(scale=2))
        if i % 150 == 0: print(f'  {tag} {i} move={mv:.3f}px {time.time()-t0:.0f}s', flush=True)
    for _ in range(45): w.append_data(fl.render(scale=2))
    w.close()
    Image.fromarray(fl.render(scale=2)).save(f'{S}/free2_{tag}.png')
    print(f'{tag} DONE final_move={mv:.3f} {time.time()-t0:.0f}s', flush=True)

rgb = np.asarray(Image.open('demoiselles.jpg')
                 .convert('RGB').resize((140,144), Image.LANCZOS))
run('demoiselles', rgb)

h,wd=100,160
ramp=np.linspace(255,0,wd).astype(np.uint8)
grad=np.stack([np.repeat(ramp[None,:],h,axis=0)]*3,-1)
scram=grad.reshape(-1,3)[np.random.default_rng(0).permutation(h*wd)].reshape(h,wd,3)
run('gradient', scram)
print('ALL DONE', flush=True)
