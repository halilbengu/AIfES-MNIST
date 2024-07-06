python get_dataset.py --train ${1:-1000} --test ${2:-100} --batch ${3:-100} --epoch ${4:-3}
rm -rf build
mkdir -p build
cd build
cmake ..
make
./AIfES_CNN