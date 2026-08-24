# Experiment Notebooks

## Final Showcases

The final notebooks all use the Task 7 sweep (`N=2,4,6,8,10`) and absolute
Gaussian noise with `sigma=0.001` on training gradients only. Validation,
random test, and fixed test gradients remain clean.

- `final_one_stage.ipynb`: direct gradient-to-mask MLP.
- `final_two_stage.ipynb`: connected gradient-to-coefficient-to-mask workflow.
- `final_three_stage.ipynb`: Task 8 Stage 1 plus the Task 9 general and specialist stages.

The previous `Old Noise` and `New Noise` experiment folders were exploratory
and have been removed. Task 1 through Task 9 educational notebooks remain in
their original folders.
