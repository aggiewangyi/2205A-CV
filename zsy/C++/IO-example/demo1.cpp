#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cctype>
#include <cstring>
#include <vector>
#include "Sales_data.h"


using namespace std;



int main(int argc, char* argv[])
{   
    cout << "argc: " << argc << endl;
    cout << "argv[0]: " << argv[0] << endl;
    cout << "argv[1]: " << argv[1] << endl;
    cout << "argv[2]: " << argv[2] << endl;


    ifstream input(argv[1]);
    ofstream output(argv[2]);

    Sales_data total;

    if (read(input, total)) {
        Sales_data trans;

        while (read(input, trans)) {
            if (total.isbn() == trans.isbn())
                total.combine(trans);
            else {
                print(output, total) << endl;
                total = trans;
            }
        }
        print(output, total) << endl;
    }
    else
        cerr << "No data?" << endl;
     
    return 0;
}