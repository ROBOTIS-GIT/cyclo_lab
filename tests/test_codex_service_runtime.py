import importlib.util
from pathlib import Path
from datetime import datetime, timedelta, timezone
from uuid import uuid4

spec=importlib.util.spec_from_file_location('bridge',Path(__file__).resolve().parents[1]/'scripts/codex-service-runtime.py')
bridge=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bridge)

def command(action,**fields):
 if action=='reset': fields.setdefault('seed',0)
 return {'command_id':str(uuid4()),'expires_at':(datetime.now(timezone.utc)+timedelta(seconds=5)).isoformat(),'command':{'action':action,**fields}}

def test_pause_step_reset_and_duplicate_are_applied_once():
 state=bridge.ControlState(); reset=[]
 pause=command('sim_pause');state.apply(pause,4,lambda **kwargs:reset.append(1));assert state.paused
 state.apply(command('step',steps=2),4,lambda:None);assert state.pending_steps==2
 state.advance(5);state.advance(6)
 value=command('reset');state.apply(value,4,lambda **kwargs:reset.append(1));state.apply(value,5,lambda **kwargs:reset.append(1));assert len(reset)==1
 assert state.results[-1]['status']=='applied'

def test_stale_or_malformed_never_changes_simulation():
 state=bridge.ControlState();item=command('sim_pause');item['expires_at']='2000-01-01T00:00:00Z';state.apply(item,1,lambda:None);assert not state.paused
 for steps in [True,0,101,'2']:
  state.apply(command('step',steps=steps),1,lambda:None)
 assert state.pending_steps==0
 assert all(x['status']=='rejected' for x in state.results)

def test_step_requires_paused_and_arbitrary_action_rejected():
 state=bridge.ControlState();state.apply(command('step',steps=1),0,lambda:None);state.apply(command('exec',code='danger'),0,lambda:None)
 assert all(x['status']=='rejected' for x in state.results)

def test_step_acknowledges_actual_completed_physics_and_rejects_overlap():
 state=bridge.ControlState();state.apply(command('sim_pause'),4,lambda:None)
 item=command('step',steps=2);state.apply(item,4,lambda:None)
 assert state.results[-1]['status']=='accepted'
 state.apply(command('step',steps=1),4,lambda:None)
 assert state.results[-1]['status']=='rejected'
 state.advance(5);assert state.results[-2]['status']=='accepted'
 state.advance(6);assert state.results[-2]['status']=='applied'
 assert state.results[-2]['applied_physics_step']==6


def test_web_view_initializes_reconnect_camera_and_directional_lights():
 from types import SimpleNamespace
 calls=[]
 scene=SimpleNamespace(add_light_directional=lambda name,**kw:calls.append(('light',name,kw)))
 server=SimpleNamespace(scene=scene,initial_camera=SimpleNamespace(),gui=SimpleNamespace(
  reset=lambda:None,configure_theme=lambda **kwargs:None))
 viewer=SimpleNamespace(_server=server,set_camera=lambda *args:calls.append(('camera',args)))
 viz=type('ViserVisualizer',(),{})();viz._viewer=viewer
 sim=SimpleNamespace(visualizers=[viz],set_camera_view=lambda *args:calls.append(('pivot',args)))
 bridge.configure_web_view(sim)
 assert calls[0][0]=='camera'
 assert calls[0][1][0]==bridge.WEB_EYE
 assert server.initial_camera.look_at==bridge.WEB_TARGET
 assert len([c for c in calls if c[0]=='light'])==2
 assert all(c[2]['cast_shadow'] is False for c in calls if c[0]=='light')
 assert calls[-1]==('pivot',(bridge.WEB_EYE,bridge.WEB_TARGET))


def test_table_migration_keeps_surface_horizontal_in_xyzw_api():
 import ast
 source=Path(__file__).resolve().parents[1]/'source/cyclo_lab/cyclo_lab/manager_based/manipulation/reach/config/omy/reach_env_cfg.py'
 tree=ast.parse(source.read_text())
 scene=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ReachSceneCfg')
 table=next(n.value for n in scene.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='table' for t in n.targets))
 state=next(k.value for k in table.keywords if k.arg=='init_state')
 x,y,z,w=ast.literal_eval(next(k.value for k in state.keywords if k.arg=='rot'))
 # In XYZW convention Rz90 maps X to Y and leaves the tabletop normal Z unchanged.
 assert abs(x)+abs(y)<1e-12
 assert abs(2*z*w-1)<1e-12
 assert abs(x*x+y*y+z*z+w*w-1)<1e-12
