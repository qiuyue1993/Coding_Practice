# Excercises
- Insertion Sort part 2
- Quicksort part 2

---
# Sorting Algorithms
## Bubble Sort 
- Compare 2 element one time and exchange them if the order is incorrect
- Time: n*n
- Space: 1 (constant)

## Selection Sort
- Select one element from inordered section and put it into ordered section
- Time: n*n
- Space: 1 (constant)

## Insertion Sort
- For every element under sorted, scan the ordered session and find a place to insert it
- Time: n*n
- Space: 1 (constant)

## Shell Sort (**！**)
- 1. choose a list t1, t2, ..., tk, ti>tj, tk=1
- 2. Implement k-times sorting
- 3. For each sorting, devide list accodring to the ti, and for every segment, implement the insertion sort
- Time: n1.3 - n*n
- Space: 1

## Merge Sort (Divide and Conquer) (**！**)
- 1. Divide a list into two sublist
- 2. For each sublist, implement merge sort
- 3. Merge two list

- Time:n*log2(n)
- Space:n

## Quick Sort 
- 1. Choose a pivot
- 2. rank based on the pivot and devide list into two section
- 3. 1-2 recursively
- Time: n*log2(n)
- Space: n*log2(n)

## Heap Sort (**！**)
- 1. Build a heap (with big root or small root) (from leaf to root)
- 2. Exchange the root with the last element 
- 3. adjust the heap (from root to leaf)
- 4. Exchange the root with the last element
- 5. Iteration
- Time: n*log2(n)
- Space: 1

## Counting Sort
- Find the largest and smallest element
- Scan the list, count for every k
- Time: n + k
- Space: n + k

## Bucket Sort
- Set a array with different value section
- Save data into different bucket
- Rank every bucket with element
- Time:n + k
- Space: n + k

## Radix Sort
- Find the largest number and the radix number
- For every radix (from low to high), gather number with same radix
- Time: n*k
- Space: n+k
