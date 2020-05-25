import os, sys, torch
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR,"../utils"))
import TDS_utils
import numpy as np

class modelnet_pc_dataset(torch.utils.data.Dataset):

  def __init__(self,directory, transforms=None,type="train"):
    super().__init__()
    self.train_fraction=0.80
    self.directory = directory
    self.type=type
    self.transforms=transforms

    folders = [dir for dir in sorted(os.listdir(directory)) if os.path.isdir(os.path.join(directory,dir))]
    class_map = {folder: i for i, folder in enumerate(folders)};
    
    self.files = []
    for class_folder in folders:
      if type=="test":
        folder=directory+'/'+class_folder+"/test"
        for file_name in os.listdir(folder):
          file_directory = folder+'/'+file_name
          self.files.append({'class':class_map[class_folder],'directory':file_directory})
      else:
        folder=directory+'/'+class_folder+"/train"
        file_names=sorted(os.listdir(folder))
        break_index=int(self.train_fraction*len(file_names))
        if type=="train":
          for i in range(0,break_index):
            file_directory = folder+'/'+file_names[i]
            self.files.append({'class':class_map[class_folder],'directory':file_directory})
        elif type=="valid":
          for i in range(break_index,len(file_names)):
            file_directory = folder+'/'+file_names[i]
            self.files.append({'class':class_map[class_folder],'directory':file_directory})

  def __getitem__(self, idx):
        dir = self.files[idx]['directory']
        y = self.files[idx]['class']
        with open(dir, 'r') as f:
            pointcloud = self.transforms((TDS_utils.read_off(f)))
        return  pointcloud, y
  
  def __len__(self):
    return len(self.files)



  def visualize_mesh(self,object=None,index=None,dir=None):
    ##visualize the original mesh data, given an object and an index, or specify a file's directory
    ## this does not distinguish between validation and training data
    if not dir:
      dir=self.directory+'/'+object
      dir+="/train/" if self.type != "test" else "/test/"
      dir=dir+os.listdir(dir)[index]
    verts,faces = TDS_utils.read_off(open(dir,'r'))

    i,j,k = np.array(faces).T
    x,y,z = np.array(verts).T
    
    TDS_utils.visualize_rotate([TDS_utils.go.Mesh3d(x=x,y=y,z=z, color='lightblue', opacity=0.50, i=i, j=j, k=k)]).show()
  
  def visualize_pointcloud(self,object=None,index=None,dir=None):
    ##visualize the random sampled point cloud data, given an object and an index, or specify a file's directory
    ## this does not distinguish between validation and training data
    if not dir:
      dir=self.directory+'/'+object
      dir+="/train/" if self.type != "test" else "/test/"
      dir=dir+os.listdir(dir)[index]
    verts,faces = TDS_utils.read_off(open(dir,'r'))
    pc=TDS_utils.PointSampler(3000)((verts,faces))
    TDS_utils.pcshow(*pc.T)
