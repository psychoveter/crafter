# Running on ray cluster

### Starting cluster
    ray up -v -y mv/train/cluster.py
    ray dashboard mv/train/cluster.py 
    

### Shutting down cluster
    ray down -v -y mv/train/cluster.py

### Running process
    ray attach -p 6006 mv/train/cluster.py
    pip uninstall opencv-python; pip install opencv-python-headless
    git clone https://github.com/psychoveter/crafter.git
    cd crafter
    git checkout mv-crafter
    python setup.py install 


