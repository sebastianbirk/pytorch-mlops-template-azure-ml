from azureml.core import Environment, Workspace
ws = Workspace.from_config() 
myenv = Environment(name="myenv")

registered_env = myenv.register(ws)
registered_env.build_local(ws)