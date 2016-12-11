Feature vector calculation function without SSE - 2.067 seconds for all sample images from the data / binary folders with SSE - 1.388. Thus obtained acceleration by approximately 33 percent. With SSE implemented image translation in grayscale, use the Sobel filter and the calculation of the gradient norm.

To run the program using the SSE must be added to the command line arguments -s, eg ./task2 -d ../../data/binary/train_labels.txt -m model.txt --train -s. To run without SSE - nothing need be added. The program displays the result of measuring the work function of time, such as SSE ON: Time [1.388] seconds.

To start the test you want to run them make test command from the root directory of the project. Test works out successfully.

Documentation is in the Doc folder.

Project was build under 32-bit ubuntu.
