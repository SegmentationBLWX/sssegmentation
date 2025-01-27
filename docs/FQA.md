# FQA


#### chainercv errors

Since the performance of all models in SSSegmentation is evaluated and reported using `chainercv`, we require the installation of this library in the requirements by default.
However, since this library has not been updated for a long time, some of the latest versions of the dependencies it requires, particularly `numpy`, are not compatible with `chainercv`.
Therefore, you might encounter installation failures at this step. In such cases, there are two possible solutions:

- The first solution is to downgrade `numpy` in your environment to version 1.x, *e.g.*, `pip install numpy==1.26.4`,
- The second solution is to manually remove `chainercv` from the requirements.txt.

It is important to note that using the second solution will involve using our custom-defined `Evaluation` class for model performance testing. 
The results may have slight differences compared to those from `chainercv`, but these differences are generally negligible.

#### scipy.interpolate.interp2d errors

Some models in SSSegmentation use the `scipy.interpolate.interp2d` function, which has been removed in SciPy 1.14.0. 
Therefore, if you encounter this situation, you need to manually downgrade your SciPy version.

### your_script.sh: line xxx: $'\r': command not found

The error typically happens when a script written on Windows (which uses `\r\n` line endings) is run on a Unix-based system (which expects `\n` line endings).
The `\r` (carriage return) character is interpreted as part of the command, leading to the command not found error.

Here's how to fix it:

- If you have dos2unix installed, you can convert the file in place: `dos2unix your_script.sh`,
- You can also use sed to remove `\r` characters: `sed -i 's/\r$//' your_script.sh`.

After converting the line endings, try running your script again. This should resolve the issue.