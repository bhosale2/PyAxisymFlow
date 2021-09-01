bind:
	@./make_project.sh
	# mv the build library files one folder backwards : this way you force recompile every time
	mv *cpython*so ../

clean:
	rm -f *cpython*so
	rm -f *bind.cpp

mplcpp:
	- g++-9 main.cpp -std=gnu++14 -Wall -I /usr/local/Cellar/python@2/2.7.15_1/Frameworks/Python.framework/Versions/2.7/include/python2.7 -I /usr/local/lib/python2.7/site-packages/numpy/core/include -L /usr/local/Cellar/python@2/2.7.15_1/Frameworks/Python.framework/Versions/2.7/lib/ -lpython2.7
