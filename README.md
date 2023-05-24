# NIDE-Graph

Create an enviroment using 
```python
pip install -r requirements.txt
```

Replace the corresponding files in the installed torchdiffeq with the files from this repository. 

To preprocess the demo dataset, run the code
```python
cd /ICEWS05-15
python ICEWS05-15_predicate_preprocess.py
```

To run experiments, run:
```python
python TANGO.py
```

The sturcture of our code:
```txt
    /source/solver.py 
        Line 352-361 shows how we serialize the MC integration.
        Line 111 shows we modeified the input of F_Func to fit MGCN.
        The rest of the .py files in /source remain unchanged.
    /models/ideblock.py     
        How we use IDESolver class similar to the basic model /models/odeblock.py.
    /models/message_passing.py   /models/MGCNLayer.py   /models/MGCN.py   
        The 2 layer Multi-relational Graph Convetional Network implemented with message passing mechanism. It is a full-batch method to update the entire embedding at once.
    /models/models.py 
        Overall network flow, calculating the dynamics of the entire time interval.
```