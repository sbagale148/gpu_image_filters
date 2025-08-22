#include <iostream>
#include <string>
#include <vector>

// Functions
int add(int a, int b) {
    return a + b;
}

long long factorial(int n) {
    long long result = 1;
    for (int i = 2; i <= n; i++) result *= i;
    return result;
}

bool is_prime(int n) {
    if (n <= 1) return false;
    for (int i = 2; i*i <= n; i++) {
        if (n % i == 0) return false;
    }
    return true;
}

int main() {
    // Hello World
    std::cout << "Hello, C++!\n";

    // Variables
    int year = 2025;
    double pi = 3.14159;
    bool enrolled = true;
    char grade = 'A';
    std::string name = "Sudama";

    std::cout << "year: " << year << "\n";
    std::cout << "pi: " << pi << "\n";
    std::cout << "enrolled: " << enrolled << "\n";
    std::cout << "grade: " << grade << "\n";
    std::cout << "name: " << name << "\n\n";

    // Loops & conditionals
    std::cout << "Counting 1 to 5 using for loop:\n";
    for (int i = 1; i <= 5; i++) {
        std::cout << i << " ";
    }
    std::cout << "\n";

    int n;
    std::cout << "Enter an integer: ";
    std::cin >> n;

    if (n < 0) std::cout << "negative\n";
    else if (n == 0) std::cout << "zero\n";
    else std::cout << "positive\n";

    if (n % 2 == 0) std::cout << n << " is even\n";
    else std::cout << n << " is odd\n\n";

    // Functions
    std::cout << "add(3, 4) = " << add(3, 4) << "\n";
    std::cout << "factorial(5) = " << factorial(5) << "\n";
    std::cout << "is_prime(7) = " << is_prime(7) << "\n\n";

    // Arrays / vectors
    std::vector<int> v = {3, 1, 4};
    std::cout << "Vector elements:\n";
    for (size_t i = 0; i < v.size(); i++) {
        std::cout << "v[" << i << "] = " << v[i] << "\n";
    }

    long long sum = 0;
    for (int x : v) sum += x;
    std::cout << "Sum of vector elements = " << sum << "\n";

    return 0;
}
