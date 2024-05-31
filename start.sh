for ((i=1; i<=5; i++))
do
    echo "Iteration $i"
    python.exe chess_crop.py
    python.exe train.py
done
