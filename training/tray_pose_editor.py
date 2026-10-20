"""Static pose editor: arm joints (deg) and tray position (m) with sliders/entries.

Usage: pal_mjlab/.venv/bin/python training/tray_pose_editor.py --pose assets/tray/carry_pose_v8.json
Save writes <pose stem>_edited.json next to the input; the last state is autosaved every 2 s
to <pose stem>_autosave.json. Visual tray only (no collision); check contacts separately.
"""
import os, json, re, time, sys, argparse
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
ROOT=Path('/home/lu/Kangaroo/Load carry task')
sys.path.insert(0,str(ROOT/'training'))
from gripper_walking_model import gripper_spec
ap=argparse.ArgumentParser();ap.add_argument('--pose',type=Path,required=True);args=ap.parse_args()
POSE=args.pose if args.pose.is_absolute() else ROOT/args.pose
pose=json.loads(POSE.read_text())
s=gripper_spec()
ground=s.worldbody.add_body(name='terrain')
ground.add_geom(name='floor',type=mujoco.mjtGeom.mjGEOM_PLANE,size=[3,3,.1],rgba=[.85,.86,.88,1])
s.add_mesh(name='tray_visual',file=str(ROOT/pose['tray_mesh']),scale=[.001]*3)
b=s.worldbody.add_body(name='tray',pos=pose['tray_pos'],quat=[.5,.5,.5,.5])
b.add_geom(name='tray_visual_geom',type=mujoco.mjtGeom.mjGEOM_MESH,meshname='tray_visual',contype=0,conaffinity=0,density=0,rgba=[.22,.55,.82,1])
s.worldbody.add_light(pos=[1,-1,3],dir=[-.3,0,-1],diffuse=[.8]*3,ambient=[.35]*3)
m=s.compile(); d=mujoco.MjData(m)
for name,value in pose['joint_positions_rad'].items():
 d.qpos[m.jnt_qposadr[m.joint(name).id]]=value
d.qpos[:7]=pose['root_qpos']
mujoco.mj_forward(m,d)
mirror=np.array([-1.,1.,-1.,1.,-1.,1.,-1.])
import tkinter as tk

ui=tk.Tk();ui.title(f"Kangaroo pose editor - {POSE.name} - {Path(pose['tray_mesh']).name}")
ui.geometry('820x750')
ui.minsize(760,750)
initial=d.qpos.copy();tray_initial=m.body_pos[m.body('tray').id].copy();variables={}
status=tk.StringVar(value='Type a value + Enter, or drag a slider. Angles: degrees.')
tk.Label(ui,textvariable=status).pack()
saved=POSE.with_name(POSE.stem+'_edited.json')
autosave=POSE.with_name(POSE.stem+'_autosave.json')
# Start each launch from the symmetric pose computed above. The saved manual pose remains
# available on disk; Reset returns to this symmetric start, while Save captures edits.
def control(parent,label,var,lo,hi,step,width):
 row=tk.Frame(parent);row.pack(fill='x',padx=5,pady=2)
 head=tk.Frame(row);head.pack(fill='x')
 tk.Label(head,text=label).pack(side='left')
 text=tk.StringVar(value=f'{var.get():.3f}')
 entry=tk.Entry(head,textvariable=text,width=10,justify='right');entry.pack(side='right')
 def sync(*_):text.set(f'{var.get():.3f}')
 def commit(event=None):
  try:
   value=float(text.get())
   if not np.isfinite(value):raise ValueError()
   if not lo<=value<=hi:
    status.set(f'{label}: allowed range {lo:.3f} to {hi:.3f}');sync();return
   var.set(value);status.set('Applied '+label)
  except ValueError:status.set('Enter a finite number.');sync()
 var.trace_add('write',sync)
 entry.bind('<Return>',commit);entry.bind('<FocusOut>',commit)
 entry.bind('<Escape>',lambda event:sync())
 tk.Scale(row,from_=lo,to=hi,resolution=step,showvalue=False,orient='horizontal',length=width,variable=var).pack(fill='x')
frame=tk.Frame(ui);frame.pack(fill='x',pady=5)
for col,side in enumerate(('left','right')):
 pane=tk.Frame(frame);pane.grid(row=0,column=col,sticky='n')
 tk.Label(pane,text=side.upper()+' ARM (degrees)').pack()
 for i in range(1,8):
  j=m.joint(f'arm_{side}_{i}_joint').id;adr=int(m.jnt_qposadr[j]);sign=1. if side=='left' else mirror[i-1];var=tk.DoubleVar(value=np.degrees(d.qpos[adr]/sign));variables[adr]=(var,sign)
  control(pane,f'arm_{side}_{i} (deg)',var,np.degrees(m.jnt_range[j,0]),np.degrees(m.jnt_range[j,1]),.01,350)
trayvars=[]
for axis in range(3):
 v=tk.DoubleVar(value=m.body_pos[m.body('tray').id,axis]);trayvars.append(v)
 control(ui,'Tray '+'XYZ'[axis]+' — WORLD / STL origin (m)',v,tray_initial[axis]-.5,tray_initial[axis]+.5,.001,700)
def save(path=saved):
 # Read controls directly so Save also captures the latest slider update.
 for adr,(var,sign) in variables.items():d.qpos[adr]=sign*np.radians(var.get())
 m.body_pos[m.body('tray').id]=[v.get() for v in trayvars]
 values={m.joint(j).name:float(d.qpos[m.jnt_qposadr[j]]) for j in range(m.njnt) if m.jnt_type[j]!=mujoco.mjtJoint.mjJNT_FREE}
 path.write_text(json.dumps({'joint_positions_rad':values,'root_qpos':d.qpos[:7].tolist(),'tray_pos':m.body_pos[m.body('tray').id].tolist(),'tray_mesh':pose['tray_mesh'],'source_pose':str(POSE),'static_only':True},indent=2))
 if path==saved:status.set(f'Pose saved to {path.name}')
def reset():
 for adr,(v,sign) in variables.items():v.set(np.degrees(initial[adr]/sign))
 for i,v in enumerate(trayvars):v.set(tray_initial[i])
buttons=tk.Frame(ui);buttons.pack();tk.Button(buttons,text='Save pose',command=save).pack(side='left');tk.Button(buttons,text='Reset pose',command=reset).pack(side='left')
with mujoco.viewer.launch_passive(m,d) as viewer:
 viewer.cam.lookat[:]=[.2,0,.95];viewer.cam.distance=2.5;viewer.cam.azimuth=135;viewer.cam.elevation=-15
 print('STATIC_TRAY_VIEWER_READY_WITH_ENTRIES_AND_SLIDERS',flush=True)
 next_save=0
 try:
  while viewer.is_running():
   ui.update()
   with viewer.lock():
    for adr,(var,sign) in variables.items():d.qpos[adr]=sign*np.radians(var.get())
    m.body_pos[m.body('tray').id]=[v.get() for v in trayvars]
    mujoco.mj_forward(m,d)
   viewer.sync()
   if time.monotonic()>next_save:
    save(autosave);next_save=time.monotonic()+2
   time.sleep(.02)
 except tk.TclError:pass
