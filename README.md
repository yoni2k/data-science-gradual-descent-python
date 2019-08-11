# Gradual Descent in Python
Main goal: implementing Gradual Descent in Python for Machine Learning.  

There is a function in form f(x,z) = w1x + w2z + b + noise.  Given inputs, and actual outputs, find w1, w2 and bias.

Seeing how various parameters effect the results.

Based on course https://www.udemy.com/the-data-science-course-complete-data-science-bootcamp

## How was done:
- Keeping the inputs constant (otherwise we'll get different results every time) - there is a function to create them that's commented out that created random results
- Inputs are read from a persistent file
- Enough inputs were generated in order to work with numerous number of observations
- Output is both to debug, and an CSV file that would be later analyzed with Excel
- Code structure:
    - External loop that calls an internal loop of gradual descent.  
    - External loop plays with different variations.
    - Internal loop performs a single Gradual Descent and stops if:
        - Maximum number of loops reached (currently 10,000)
        - Loss is not improving (good result - we have reached the solution)
        - Loss is becoming larger - we are reaching for the infinity, no point to continue

## Artifacts:
- inputs.txt - input file
- output.txt - last output file, as long as inputs and code will not be changed, this is the file that will be generated
- gradual_descent.py - main python file
- requirements.txt - file with Python package requirements
- outputs conclusions folder - has different outputs with some formatting and conclusions

## Variations - to see how each effects the result
- 2 functions to find:
    - 2x - 3z + 5
    - 13x + 7z - 12
- Different loss functions:
    - np.sum(deltas ** 2) / 2
    - np.sum(abs(deltas))
- Different number of observations: 1,000 and 10,000
- Different learning rates: 0.00000001, 0.0000001, 0.000001, 0.00001
- Different starting weights both for xs, zs and biases: -10, -0.1, 0.1, 10

## Conclusions:
- A great measure of comparison between different runs is **Loss / number of observations**.  It accurately captures how far we are from reaching the result, and can be used to compare between different runs of Gradual Descent, and to predict whether the method converged.
- 2 different functions to be found behave in a very similar way, same conclusion is reached in both functions together. The only exception to the rule being specific weights and specific number of iteratoins (see below)
- 2 different loss functions reach similar results (go to infinity together, reach good result together), with the difference being number of iterations done
- Different number of observations: the more observations, less iterations are needed to reach the result
- Learning rates: low learning rate is slower (more iterations), but too high of a rate causes the gradual descent to shoop for infinity.  Therefore best to have highest learning rate that doesn't shoot for infinity.  It was observed that having learning rate x 10, means having around / 10 of iterations. 
- Different starting weights slightly affect the number of iterations, but not the end result.
- Weight of bias is extremely sensitive (depending where we start), we might reach the result or not within a given number of iterations.  However, didn't see a difference for number iterations based on weights chosen for the xs and zs.
- Bias is resolved last, there are cases when weights for xs and zs are OK, but bias is not yet.
- Weights (of bias) effect the number of iteration, with initial weight closer to the actual weight taking less iterations
- Many loops or observations makes the descend much slower

## Warning:
The internal loop was stopped when loss stopped improving greatly, or started becoming higher (shooting for infinity).  However, it was explained in the lectures, that loss is not linear, it could go up and down.  So it might be OK for the Linear function we were exploring, but not in general.  In general - we need to give enough iterations.
