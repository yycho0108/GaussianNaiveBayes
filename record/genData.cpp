#include <iostream>
#include <random>
#include <functional>
#include <ctime>

auto f = std::bind(std::uniform_real_distribution<double>(0.0,1.0), std::default_random_engine(time(0)));

int main(int argc, char* argv[]){
	int range = 100;
	if (argc > 1){
		range = std::atoi(argv[1]);
	}
	for(int i=0;i<range;++i){
		int lhs = std::round(f());
		int rhs = std::round(f());
		std::cout << lhs << ',' << rhs << ',' << (lhs^rhs) << std::endl;
	}
}
