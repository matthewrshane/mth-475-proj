using PythonCall;

py_sys = pyimport("sys")
py_sys.path.append(pwd())

rml = pyimport("rakesh_ml")
print(rml.predict(1))