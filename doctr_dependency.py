import subprocess

def install_doctr():
    try:
        import tensorflow
        command = "pip install python-doctr[tf]"
    except:
        try: 
            import torch
            command = "pip install python-doctr[torch]"
        except:
            raise ModuleNotFoundError("Please install either Tensorflow or Torch")
   
    res = subprocess.run(command,shell=True,capture_output = False)
    
    
def install_fastapi():
    command = "pip install fastapi"
    subprocess.run(command,shell=True,capture_output = False)
    command2 = 'pip install "uvicorn[standard]"'
    subprocess.run(command2,shell=True,capture_output=False)
    