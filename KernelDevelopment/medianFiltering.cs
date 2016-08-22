using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Diagnostics;
using Cudafy;
using Cudafy.Host;
using Cudafy.Translator;

namespace KernelDevelopment
{
    class medianFiltering
    {
        /*
         * Bug tested with continously increasing values.
         * TODO: test with decreasing values and check possible optimization.
         */ 
        public static void Execute()
        {
            CudafyModule km = CudafyTranslator.Cudafy();
            km.GenerateDebug = true;
            GPGPU gpu = CudafyHost.GetDevice(CudafyModes.Target, CudafyModes.DeviceId);
            gpu.LoadModule(km);

            // cudart.dll must be accessible!
            GPGPUProperties prop = null;
            try
            {
                prop = gpu.GetDeviceProperties(true);
            }
            catch (DllNotFoundException)
            {
                prop = gpu.GetDeviceProperties(false);
            }
            // Setup
            int width           = 50;                                       // filter window width.
            int depth           = 10000;                                    // z.
            int framewidth      = 64;
            int frameheight     = 64;
            int N               = depth * framewidth * frameheight;         // size of entry.
            int[] meanVector    = medianFiltering.generateTest(depth);      // frame mean value.
            int[] test_data     = medianFiltering.generateTest(N, depth);   // pixel values organized as: x1y1z1, x1y1z2,...,x1y1zn, x2y1z1,...                                                               
            // Profiling:
            Stopwatch watch     = new Stopwatch();
            watch.Start();

            // Transfer data to device.
            int[] device_data       = gpu.CopyToDevice(test_data);
            int[] device_meanVector = gpu.CopyToDevice(meanVector);
            int[] device_result     = gpu.Allocate<int>(N);
            int[] device_window     = gpu.Allocate<int>((2 * width + 1) * framewidth * frameheight);

            // Run kernel.

            gpu.Launch(new dim3(framewidth, frameheight), 1).medianKernel(width, device_window, depth, device_data, device_meanVector, device_result);
            
            // Collect results.
            int[] result = new int[N];
            gpu.CopyFromDevice(device_result, result);

            //Profile:
            watch.Stop();
            Console.WriteLine("compute time: " + watch.ElapsedMilliseconds);

            // Check some outputs:
            //   Console.WriteLine("Start: " + result[0]);
            int start =0*depth;
            for (int i = 50+start; i < depth*10; i += depth)
            {
                Console.WriteLine("Row# " + (i + 1 -start) + ": " + result[i]);
            }
            

            // Clear gpu.
            gpu.FreeAll();
            gpu.HostFreeAll();
            Console.ReadKey(); // keep console up.

        }

        [Cudafy]
        /*
         * kernel takes median width, swapvector that is (2*window_width+1)*x_width*y_height, z_depth, vector with input data organized as:
         *  x1y1z1, x1y1z2,...,x1y1zn, x2y1z1,...
         *  mean value vector for each frame (z1, z2,...)
         *  output vector of same size as input data vector.
         *  
         *  Output is: Input value - frameMean*median.
         * 
         */
        public static void medianKernel(GThread thread, int windowWidth, int[] filterWindow, int depth, int[] inputVector, int[] meanVector, int[] answer)
        {
            //          int y   = thread.blockIdx.x;
            //          int x   = thread.threadIdx.x;
            //          int idx = x + (y * thread.blockDim.x);          // which pixel.     

            int y = thread.blockIdx.y;
            int x = thread.blockIdx.x;
            int idx = x + (y * thread.gridDim.x);          // which pixel.  
           
           if (idx < inputVector.Length / depth)           // if pixel is included.
            {
                int zSlice = 0;            // Used to keep track of zslice, used in generating final result values.
                int answerIndex = idx * depth; // Where to start placing results.
                int filterIndex = idx * (2 * windowWidth + 1); // keep track of filter window insertion.
                Boolean isOdd = true;            // Keep track of if current effective filter window is odd or even, for main portion this is always odd.
                int temp;       // Swap variable for shifting filterWindow entries.
                
                // Start populating filterWindow with windowWidth first number of values from inputVector:
                for (int populateIDX = idx * depth; populateIDX < idx * depth + windowWidth+1; populateIDX++)
                {
                    filterWindow[filterIndex] = inputVector[populateIDX];
                    filterIndex++;
                }
                
                // using bubblesort, sort the windowWidth first number of elements:

                int index = idx * (2 * windowWidth + 1); // start point
                
                for (int bubbleSortOuter = index + 1; bubbleSortOuter < index + windowWidth+1; bubbleSortOuter++)
                {
                    for (int bubbleSortInner = index; bubbleSortInner < index + windowWidth - bubbleSortOuter; bubbleSortInner++)
                    {
                        if (filterWindow[bubbleSortInner] > filterWindow[bubbleSortInner + 1])
                        {
                            temp = filterWindow[bubbleSortInner];
                            filterWindow[bubbleSortInner] = filterWindow[bubbleSortInner + 1];
                            filterWindow[bubbleSortInner + 1] = temp;
                        }
                    }
                }
                if (windowWidth % 2 == 0) // is odd?
                {
          //          answer[answerIndex] = inputVector[answerIndex] - meanVector[zSlice] * filterWindow[index + (windowWidth - 1) / 2]; // InputVector - meanvalueFrame*median.
                    answer[answerIndex] = filterWindow[index + (windowWidth) / 2]; // InputVector - meanvalueFrame*median.
                    isOdd = false;
                }
                else
                {
            //        answer[answerIndex] = inputVector[answerIndex] - meanVector[zSlice] * (filterWindow[index + (windowWidth) / 2] + filterWindow[index + ((windowWidth) / 2) - 1]) / 2; // InputVector - meanvalueFrame*median.
                    answer[answerIndex] = (int)Math.Ceiling((filterWindow[index + (windowWidth-1) / 2] + filterWindow[index + ((windowWidth-1) / 2) + 1]) / 2.0); // InputVector - meanvalueFrame*median. (int) 2+3/2.0 = 2 not 3.                  
                    isOdd = true;
                }

                // Update counters:
                zSlice++; // index for meanVector.
                if (isOdd)
                    index = idx * (2 * windowWidth + 1) + (windowWidth - 1) / 2 + 1;  // start index for filterWindow.
                else
                    index = idx * (2 * windowWidth + 1) + windowWidth/2 + 1;  // start index for filterWindow.
                answerIndex++; // Index of entry to calculate median for.
               
                for (int populateIndex = idx * depth + windowWidth+1; populateIndex < idx * depth + 2 * windowWidth+1; populateIndex++) // Add one element at a time until 2xwindowWidth+1 elements are in the list.
                {
                    filterWindow[filterIndex] = inputVector[populateIndex];
                    // Bubblesort filterWindow, not pretty but easy to implement:
                    for (int bubbleSortOuter = idx * (2 * windowWidth + 1) + 1; bubbleSortOuter < filterIndex; bubbleSortOuter++) // Loop over current filterWindow.
                    {
                        for (int bubbleSortInner = idx * (2 * windowWidth + 1); bubbleSortInner < filterIndex - bubbleSortOuter; bubbleSortInner++)
                        {
                            if (filterWindow[bubbleSortInner] > filterWindow[bubbleSortInner + 1])
                            {
                                temp = filterWindow[bubbleSortInner];
                                filterWindow[bubbleSortInner] = filterWindow[bubbleSortInner + 1];
                                filterWindow[bubbleSortInner + 1] = temp;
                            }
                        }
                    }

                    if (isOdd) // is odd?
                    {
   //                     answer[answerIndex] = inputVector[answerIndex] - meanVector[zSlice] * filterWindow[index ]; // InputVector - meanvalueFrame*median.
                        answer[answerIndex] = filterWindow[index]; // InputVector - meanvalueFrame*median.
                        isOdd = false;                       
                        index++;        // start index for filterWindow.
                    }
                    else
                    {
     //                   answer[answerIndex] = inputVector[answerIndex] - meanVector[zSlice] * (filterWindow[index ] + filterWindow[index - 1]) / 2; // InputVector - meanvalueFrame*median.
                        answer[answerIndex] = (int)Math.Ceiling((filterWindow[index] + filterWindow[index - 1]) / 2.0 ); // InputVector - meanvalueFrame*median. (int) 2+3/2.0 = 2 not 3.
                
                        isOdd = true;
                    }

                    // Update counters:
                    zSlice++;       // index for meanVector.
                    filterIndex++;  // Insertion index for filterWindow.                    
                    answerIndex++; // Index of entry to calculate median for.
                }


                // filterWindow now fully populated.
                // find oldest entry and replace with newest. run through sorting algorithm on this entry and upwards.
                // Loop through centre portion of input data, changing algorithm once end has been reached.                
                filterIndex = idx * (2 * windowWidth + 1); // start index for this coordinate.
                index = filterIndex + windowWidth; // Median element.          
                int upperWindowBound = index + windowWidth; // correct for filter window.              
                Boolean found = false;
                int searchCounter = 0;

                while (answerIndex < (idx + 1) * depth - windowWidth) // until end part has been reached:
                {
                    // find oldest entry and replace with newest.
                    found           = false;           // Update success counter.
                    searchCounter   = filterIndex; // starting index to search in filterWindow.
                    while (!found && searchCounter < upperWindowBound+1)
                    {
                        if (filterWindow[searchCounter] == inputVector[answerIndex - windowWidth - 1]) // find entry in filterWindow matching oldest entry.
                        {
                            found = true;
                            filterWindow[searchCounter] = inputVector[answerIndex + windowWidth]; // replace oldest entry with the next one in line.

                            // Shift filterWindow so that it is ordered again. TODO: Check which order occurs most often. 
                            if (searchCounter == filterIndex && filterWindow[searchCounter] < filterWindow[searchCounter + 1]) // If we just replaced the first element and it is the smallest.
                            {
                                // do nothing, list sorted.
                            }
                            else if (searchCounter == upperWindowBound && filterWindow[searchCounter] > filterWindow[searchCounter - 1]) // If we just replaced the last element and it is the largest.
                            {
                                // do nothing, list sorted.
                            }
                            else if (searchCounter == filterIndex && filterWindow[searchCounter] > filterWindow[searchCounter + 1]) // If we just replaced the first element and it is the larger then the second.
                            {
                                
                                while (filterWindow[searchCounter] > filterWindow[searchCounter + 1] && searchCounter < upperWindowBound) // whilst the new entry is larger then the next entry and we're not at the end.
                                {
                                    temp = filterWindow[searchCounter + 1];
                                    filterWindow[searchCounter + 1] = filterWindow[searchCounter];
                                    filterWindow[searchCounter] = temp;
                                    searchCounter++;
                                }
                                
                            }
                            else if (searchCounter == upperWindowBound && filterWindow[searchCounter] < filterWindow[searchCounter - 1]) // If we just replaced the last element and it is the smaller then the second last.
                            {
                                
                                while (filterWindow[searchCounter] < filterWindow[searchCounter - 1] && searchCounter > filterIndex) // whilst the new entry is smaller then the preceeeding entry and we're not at the start.
                                {
                                    temp = filterWindow[searchCounter - 1];
                                    filterWindow[searchCounter - 1] = filterWindow[searchCounter];
                                    filterWindow[searchCounter] = temp;
                                    searchCounter++;
                                }
                            }
                            else if (filterWindow[searchCounter] == filterWindow[searchCounter + 1] || filterWindow[searchCounter] == filterWindow[searchCounter - 1]) // if we just placed a new element equal to one of its neighbours.
                            {
                                // Do nothing, list sorted.
                            }
                            else
                            {
                                if (filterWindow[searchCounter] > filterWindow[searchCounter + 1]) // if we should shift the new entry upwards.
                                {                                  
                                    while (filterWindow[searchCounter] > filterWindow[searchCounter + 1] && searchCounter < upperWindowBound) // whilst the new entry is larger then the next entry and we're not at the end.
                                    {
                                        temp = filterWindow[searchCounter + 1];
                                        filterWindow[searchCounter + 1] = filterWindow[searchCounter];
                                        filterWindow[searchCounter] = temp;
                                        searchCounter++;
                                    }
                                }
                                else // shift new element down.
                                {
                                    
                                    while (filterWindow[searchCounter] < filterWindow[searchCounter - 1] && searchCounter > filterIndex) // whilst the new entry is smaller then the preceeeding entry and we're not at the start.
                                    {
                                        temp = filterWindow[searchCounter - 1];
                                        filterWindow[searchCounter - 1] = filterWindow[searchCounter];
                                        filterWindow[searchCounter] = temp;
                                        searchCounter++;
                                    }

                                }

                            }

                        } // If correct value has been found.
                        searchCounter++;
                    } // searching while loop.                
                      //answer[answerIndex] = inputVector[answerIndex] - meanVector[zSlice] * filterWindow[index]; // InputVector - meanvalueFrame*median.
                    answer[answerIndex] = filterWindow[index]; // InputVector - meanvalueFrame*median.
                    zSlice++;       // index for meanVector.                    
                    answerIndex++; // Index of entry to calculate median for.
                }// main while loop.

                isOdd = false;  // full filter window is always odd number long (2W+1), we start by reducing index by 1.

                int maxFilterIdx = index + windowWidth; // last index for current filterwindow.
          
                while (answerIndex < (idx + 1) * depth) // loop over remaining entries.
                {      
                    if (isOdd)
                    {
  //                      answer[answerIndex] = inputVector[answerIndex] - meanVector[zSlice] * filterWindow[index]; // InputVector - meanvalueFrame*median.
                        answer[answerIndex] = filterWindow[index]; // InputVector - meanvalueFrame*median.
                        isOdd               = false;
                    }
                    else
                    {
 //                       answer[answerIndex] = inputVector[answerIndex] - meanVector[zSlice] * (filterWindow[index] + filterWindow[index - 1]) / 2; // InputVector - meanvalueFrame*median.
                        answer[answerIndex] = (int) Math.Ceiling((filterWindow[index] + filterWindow[index + 1]) / 2.0); // InputVector - meanvalueFrame*median.
                        isOdd               = true;
                        index++;
                    }
                    // Update counters:
                    zSlice++;       // index for meanVector.                    
                    answerIndex++;  // Index of entry to calculate median for.                        
                } // loop over remaining entries.
            } // end main check of idx.
        } // End medianKernel.

        public static int[] generateTest(int N) // generate test vector to simulate mean frame values.
        {
            int[] test = new int[N];
            for (int i = 0; i < test.Length; i++)
            {
                test[i] = 1;
            }
            return test;
        }

        public static int[] generateTest(int N, int depth) // Generate test vector to easily see if median calculations are correct.
        {
            int[] test = new int[N];
            int count = 0;
            for (int i = 0; i < test.Length; i++)
            {
                test[i] = count;
                count++;
                if (count == depth)
                    count = 0;
            }
            return test;
        }
    }
}
