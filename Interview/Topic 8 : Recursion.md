# Excercises
- Fibonacci Numbers

---
# Notes
## Three important element
- Boundary condition
- Recursive advance session
- Recursive return session
- While the boundary condition is satisfied, the function returns; If not, the function advance

## Recursion and loop 
- Usually loop is more efficient than recursion
- Turn a recursion to a loop using stack structure

*Steps*
- When a function is called, stack the parameters and return position into a stack
- The function is accessible to the stack
- When return a function, find back position through the stack, and return all values and parameters, finally delete them
