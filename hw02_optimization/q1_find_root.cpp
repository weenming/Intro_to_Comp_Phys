#include <iostream>
#include <iomanip>
using namespace std;

double func0(double x) {
    double f = x * x * x -5 * x + 3;
    return f;
}

double d_func0(double x) {
    return 3 * x * 2 - 5;
}

void bisection(double* bracket) {
    // input: func(bracket[0])>0 and func(bracket[1])<0
    double bisectioned = (bracket[0] + bracket[1]) / 2;
    if(func(bisectioned) > 0){
        bracket[0] = bisectioned;
    }
    else if(func(bisectioned) < 0){
        bracket[1] = bisectioned;
    }
    else{
        bracket[0] = bisectioned;
        bracket[1] = bisectioned;
    }
}

double newton(double x) {  
    return -func(x)/d_func(x);
}

void print_sol (double sol, double bias) {
    cout << setprecision(20) << sol << "+-" << bias / 2 << endl;
}


class FindSolBis{
public:
    FindSolBis(double x_posf, double x_negf, double (*func_in)(double)): pos(x_posf), neg(x_negf), func(func_in) {};
    void init_df(double (*func_in)(double)) {d_func = func_in;}// use analytical derivative
    void find();
    void print_sol();
protected:
    // f(pos) > 0 and f(neg) < 0
    double pos;
    double neg;
    double (*func) (double);
    double (*d_func) (double);
    double num_df(double);// if derivative is not given explicitly, use numerical method to calculate df/dx
};

double FindSolBis::num_df(double x) {
    double dx = 1e-5;
    return (func(x+dx) - func(x)) / dx;
}

void FindSolBis::find() {
    double new_mid;
    while(abs(pos - neg) > 1e-4) {
        new_mid = func((pos + neg) / 2);
        if(new_mid > 0) {
            pos = new_mid;
        }
        else if(new_mid < 0) {
            neg = new_mid;
        }
        else{
            pos = new_mid;
            neg = new_mid;
        }
    }
}

int main() {

    system("pause");
    return 0;
}