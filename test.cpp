#include <boost/scoped_array.hpp>
#include <iostream>
#define BOOST_PYTHON_STATIC_LIB
#define BOOST_LIB_NAME "boost_numpy39"
#include <boost/config/auto_link.hpp>
#include <boost/python/numpy.hpp>

using namespace std;
namespace p = boost::python;
namespace np = boost::python::numpy;


// Actual C++/C code here that does everything
void count_and_sum(double* array, int num, double output[]) {
    double sum = 0.0;

    for (int i = 0; i < num; ++i) {
        sum += array[i];
    }

    // Set output values
    output[0] = sum;
    output[1] = (double)num;
}

// A few translators here to ensure that numpy datatypes convert to pointers and what not
np::ndarray wrap_count_and_sum(np::ndarray const & array) {

    // Make sure we get doubles
    if (array.get_dtype() != np::dtype::get_builtin<double>()) {
        PyErr_SetString(PyExc_TypeError, "Incorrect array data type");
        p::throw_error_already_set();
    }

    // Could also pass back a vector, but unsure if you use C++ or C
    static double output[2]; // the static here is important, keeps it around!
    count_and_sum(reinterpret_cast<double*>(array.get_data()), array.shape(0), output);


    // Turning the output into a numpy array
    np::dtype dt = np::dtype::get_builtin<double>();
    p::tuple shape = p::make_tuple(2); // It has shape (2,)
    p::tuple stride = p::make_tuple(sizeof(double)); // 1D array, so its just size of double
    np::ndarray result = np::from_data(output, dt, shape, stride, p::object());
    return result;

}

// Deciding what to expose in the library python can import
BOOST_PYTHON_MODULE(cuda_python) {  // Thing in brackets should match output library name
    Py_Initialize();
    np::initialize();
    p::def("count_and_sum", wrap_count_and_sum);
}