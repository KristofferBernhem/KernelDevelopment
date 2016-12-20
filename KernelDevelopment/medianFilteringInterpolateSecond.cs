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
    class medianFilteringInterpolateSecond
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
            int[] answer = {604, 810, 789, 753, 653, 717, 619, 736, 571, 688, 720, 804, 906, 663, 674, 883, 740, 754, 658, 599
                          , 869, 689, 616, 746, 625, 676, 765, 639, 689, 713, 674, 659, 697, 822, 693, 638, 759, 750, 706, 720, 
                            728, 672, 530, 543, 675, 797, 666, 916, 615, 725, 537, 588, 625, 715, 757, 786, 608, 650, 687, 774};
            int width           = 50;                                       // filter window width.
            int depth = 2499;// answer.Length;                                    // z.
            int framewidth      = 2;
            int frameheight     = 2;
            int N               = depth * framewidth * frameheight;         // size of entry.
            float[] meanVector = medianFilteringInterpolateSecond.generateTest(depth);      // frame mean value.
            float[] test_data =  medianFilteringInterpolateSecond.generateTest(N, depth);   // pixel values organized as: x1y1z1, x1y1z2,...,x1y1zn, x2y1z1,...                                                               
            // Profiling:
            Stopwatch watch     = new Stopwatch();
            watch.Start();

            // Transfer data to device.
            float[] device_data       = gpu.CopyToDevice(test_data);
            float[] device_meanVector = gpu.CopyToDevice(meanVector);
            int[] device_result       = gpu.Allocate<int>(N);
            float[] device_window     = gpu.Allocate<float>((2 * width + 1) * framewidth * frameheight);

            // Run kernel.

            gpu.Launch(new dim3(framewidth, frameheight,1), 1).medianKernel(width, device_window, depth, device_data, device_meanVector, 1, device_result);            

            int[] result2 = new int[N];
            gpu.CopyFromDevice(device_result, result2);
            int i = 0;
            Console.Out.WriteLine("First");
           for (int h = 0; h < framewidth * frameheight * 60; h += framewidth * frameheight)
             //  for (int h = 0; h < 101; h++ )
                {
                    //                Console.Out.WriteLine((result[h] - result2[h]));
                    
                    if (i == 10)
                    //if(h%10 == 0)
                    {
                        Console.Out.WriteLine("");
                        i = 0;
                    }
                    Console.Out.Write(result2[h] + " ");
                    i++;
                }
           Console.Out.WriteLine("");
           Console.Out.WriteLine("Second");
           i = 0;
           for (int h = 1; h < framewidth * frameheight * 60; h += framewidth * frameheight)
           //  for (int h = 0; h < 101; h++ )
           {
               //                Console.Out.WriteLine((result[h] - result2[h]));

               if (i == 10)
               //if(h%10 == 0)
               {
                   Console.Out.WriteLine("");
                   i = 0;
               }
               Console.Out.Write(result2[h] + " ");
               i++;
           }
           Console.Out.WriteLine("");
           Console.Out.WriteLine("Third");
           for (int h = 2; h < framewidth * frameheight * 60; h += framewidth * frameheight)
           //  for (int h = 0; h < 101; h++ )
           {
               //                Console.Out.WriteLine((result[h] - result2[h]));

               if (i == 10)
               //if(h%10 == 0)
               {
                   Console.Out.WriteLine("");
                   i = 0;
               }
               Console.Out.Write(result2[h] + " ");
               i++;
           }
           Console.Out.WriteLine("");
           Console.Out.WriteLine("Fourth");
           for (int h = 3; h < framewidth * frameheight * 60; h += framewidth * frameheight)
           //  for (int h = 0; h < 101; h++ )
           {
               //                Console.Out.WriteLine((result[h] - result2[h]));

               if (i == 10)
               //if(h%10 == 0)
               {
                   Console.Out.WriteLine("");
                   i = 0;
               }
               Console.Out.Write(result2[h] + " ");
               i++;
           }
            /*
           Console.Out.WriteLine("");
           Console.Out.WriteLine("");
             for (int h = 0; h < 101; h++ )
           {
               //                Console.Out.WriteLine((result[h] - result2[h]));

               if(h%10 == 0)
               {
                   Console.Out.WriteLine("");
                   i = 0;
               }
               Console.Out.Write(result2[h] + " ");
               i++;
           }*/
           Console.Out.WriteLine("");
           Console.Out.WriteLine("");
                //Profile:
                watch.Stop();                
            Console.WriteLine("compute time: " + watch.ElapsedMilliseconds);
            
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
         *  This version skips nStep entries and interpolates results.
         *  Output is: Input value - frameMean*median.
         *
         */
        public static void medianKernel(GThread thread, int windowWidth, float[] filterWindow, int depth, float[] inputVector, float[] meanVector, int nStep, int[] answer)
        {
            int y = thread.blockIdx.y;
            int x = thread.blockIdx.x;
            int idx = x + (y * thread.gridDim.x);          // which pixel.              
            if (idx < inputVector.Length / depth)           // if pixel is included.
            {
                int low = idx * (2 * windowWidth + 1);
                int high = low + windowWidth;
                float temp = 0;
                int zSlize = 0;
                int inpIdx = idx*depth; // variable to keep track of latest added inputvector entry.
                int filtIdx = low; // variable to keep track of next position to add element to.
                int outIdx = idx;
                int frameSize = (inputVector.Length / depth);
                float lastMedian = 0;
                float interpolationStepSize = 0;
                while (inpIdx < (idx+1)* depth) // normalize input based on provided meanVector.
                {
                    inputVector[inpIdx] = inputVector[inpIdx] / meanVector[zSlize];           
                    inpIdx++;
                    zSlize++;            
                }
              
                zSlize = 0;           // reset.
                inpIdx = idx * depth; // reset.
                while (filtIdx <= high) // populate first windowWidth+1 elements.
                {
                    filterWindow[filtIdx] = inputVector[inpIdx];
                    filtIdx++;
                    inpIdx += nStep; // load every nStep entry of inputVector
                }


                high++;
                Boolean swapped = true;
                int j = 0;
                while(swapped)
                {
                    swapped = false;
                    j++;
                    for (int i = low; i < high-j; i++)
                    {
                        if (filterWindow[i] > filterWindow[i+1])
                        {
                            temp = filterWindow[i];
                            filterWindow[i] = filterWindow[i + 1];
                            filterWindow[i + 1] = temp;
                            swapped = true;
                        }
                    }
                }
                /*
                 * filterWindow sorted for current entries.
                 */


                high = windowWidth+1;

                inpIdx = idx * depth;
                answer[outIdx] = (int)(meanVector[zSlize] * (inputVector[inpIdx] - filterWindow[low + (high / 2)])); // first entry.
                lastMedian = filterWindow[low + (high / 2)];
                if (answer[outIdx] < 0)
                    answer[outIdx] = 0;


                
                outIdx += frameSize*nStep; // next save step.
                inpIdx += nStep; // load every nStep entry of inputVector                
                zSlize += nStep;
                while (filtIdx < low+2*windowWidth+1) // fully populate filterWindow.
                {
                    filterWindow[filtIdx] = inputVector[inpIdx+(windowWidth)*nStep];
                    swapped = false;
                    j = filtIdx;
                    while(!swapped) // sort the new entry.
                    {
                        if (filterWindow[j - 1] > filterWindow[j])
                        {
                            temp = filterWindow[j];
                            filterWindow[j] = filterWindow[j - 1];
                            filterWindow[j - 1] = temp;
                        }
                        else
                            swapped = true;
                        j--;
                        if (j == low) // if wer're on the first index, dont sort the new entry out of bounds.
                            swapped = true;
                    } // done sorting in the new entry.

                    answer[outIdx] = (int)(meanVector[zSlize]*(inputVector[inpIdx]-
                        (filterWindow[low +  + high / 2] +  
                        (high%2)* filterWindow[low + high%2 + high / 2])/(1+(high%2))));

                    if (answer[outIdx] < 0)
                        answer[outIdx] = 0;
                    interpolationStepSize = (filterWindow[low +  + high / 2] +  
                        (high%2)* filterWindow[low + high%2 + high / 2])/(1+(high%2)) - 
                        lastMedian; // current - last median value.
                    interpolationStepSize /= nStep;
                    for (int i = 1; i < nStep; i++ )
                    {
                        answer[outIdx - nStep*frameSize + i*frameSize] = 
                            (int) (meanVector[zSlize-nStep+i]*(
                            inputVector[inpIdx - nStep + i] -
                            lastMedian - interpolationStepSize*i
                            ));
                        if (answer[outIdx - nStep * frameSize + i * frameSize] < 0)
                            answer[outIdx - nStep * frameSize + i * frameSize] = 0;
                    }
                    lastMedian = (filterWindow[low + +high / 2] + 
                        (high % 2) * filterWindow[low + high % 2 + high / 2]) / (1 + (high % 2)); // median for this round, store for next.


                    high++;
                    inpIdx += nStep; // load every nStep entry of inputVector
                    filtIdx++; //update counter on position to insert next entry to.
                    outIdx += frameSize * nStep; // next save step.
                    zSlize += nStep;
                }
                
                while (inpIdx < (idx + 1) * depth - (windowWidth) * nStep) // main loop over the middle part of the vector.
                {
                  //  inputVector[inpIdx - (windowWidth + 1) * nStep]; // entry to be replaced.;
       
                    j = low;
                    swapped = false;
                    while (!swapped)
                    {
                        if (inputVector[inpIdx - (windowWidth +1) * nStep] == filterWindow[j]) 
                        {
                            filterWindow[j] = inputVector[inpIdx+(windowWidth)*nStep]; // replace oldest entry with the next.
                            swapped = true;
                        }
                        else
                            j++;
                    } // entry replaced. Time to sort the new list.

               
                    swapped = false;
                    if (j == low) // if we replaced the first entry in filterWindow
                    {

                    } // j==low
                    else if (j == low + 2*windowWidth) // if we replaced the last entry in filterwindow
                    {

                    } // j == low + 2*windowWidth+1
                    else
                    {
                        if (filterWindow[j] < filterWindow[j+1] && 
                            filterWindow[j] > filterWindow[j-1]) // if already sorted
                        {
                            // do nothing.
                        }
                        else if (filterWindow[j] > filterWindow[j+1]) // if the new entry needs to be shifted up:
                        {
                            while(!swapped)
                            {
                                temp = filterWindow[j + 1];
                                filterWindow[j + 1] = filterWindow[j];
                                filterWindow[j] = temp;
                                j++;
                                if (filterWindow[j] < filterWindow[j+1])
                                {
                                    swapped = true;
                                }
                                if (j == low + 2*windowWidth)
                                {
                                    swapped = true;
                                }
                            }
                        } // filterWindow[j] > filterWindow[j+1]
                        else if (filterWindow[j] < filterWindow[j - 1]) // if the new entry needs to be shifted up:
                        {
                            while (!swapped)
                            {
                                temp = filterWindow[j - 1];
                                filterWindow[j - 1] = filterWindow[j];
                                filterWindow[j] = temp;
                                j--;
                                if (filterWindow[j] > filterWindow[j - 1])
                                {
                                    swapped = true;
                                }
                                else if (j == low)
                                {
                                    swapped = true;
                                }
                            }
                        } // filterWindow[j] < filterWindow[j-1]
                    } // if j is in the center portion.
                    // new entry now sorted, calculate median and interpolate:
                    
                    answer[outIdx] = (int)(meanVector[zSlize] * (inputVector[inpIdx] -
                        (filterWindow[low + +high / 2] )));
                    if (answer[outIdx] < 0)
                        answer[outIdx] = 0;

                    interpolationStepSize = filterWindow[low + +high / 2] -
                        lastMedian; // current - last median value.
                    interpolationStepSize /= nStep;
                    for (int i = 1; i < nStep; i++)
                    {
                        answer[outIdx - nStep * frameSize + i * frameSize] =
                            (int)(meanVector[zSlize - nStep + i] * (
                            inputVector[inpIdx - nStep + i] -
                            lastMedian - interpolationStepSize * i
                            ));
                        if (answer[outIdx - nStep * frameSize + i * frameSize] < 0)
                            answer[outIdx - nStep * frameSize + i * frameSize] = 0;
                    }
                    lastMedian = filterWindow[low + +high / 2]; // median for this round, store for next.
                    
                    
          //          high++;
                    inpIdx += nStep; // load every nStep entry of inputVector
            //        filtIdx++; //update counter on position to insert next entry to.
                    outIdx += frameSize * nStep; // next save step.
                    zSlize += nStep;

                } // main loop.
                
                
                high--;
                while (inpIdx < (idx + 1) * depth)
                {
                    //  inputVector[inpIdx - (windowWidth + 1) * nStep]; // entry to be removed;
                    j = low;
                    swapped = false;
                    while (!swapped)
                    {
                        if (inputVector[inpIdx - (windowWidth + 1) * nStep] == filterWindow[j]) 
                        {
                            while (j < low + high) // if this is not the last entry.
                            {
                                temp = filterWindow[j + 1];
                                filterWindow[j + 1] = filterWindow[j];
                                filterWindow[j] = temp;
                                j++;
                            }
                                
                            swapped = true;
                        }
                        else
                            j++;
                    } // entry replaced. Time to sort the new list.
                    answer[outIdx] = (int)(meanVector[zSlize] * (inputVector[inpIdx] -
                      (filterWindow[low + +high / 2] +
                        (high % 2) * filterWindow[low + high % 2 + high / 2]) / (1 + (high % 2))));
                    if (answer[outIdx] < 0)
                        answer[outIdx] = 0;

                    interpolationStepSize = (filterWindow[low + +high / 2] +
                        (high % 2) * filterWindow[low + high % 2 + high / 2]) / (1 + (high % 2)) -
                        lastMedian; // current - last median value.
                    interpolationStepSize /= nStep;
                    for (int i = 1; i < nStep; i++)
                    {
                        answer[outIdx - nStep * frameSize + i * frameSize] =
                            (int)(meanVector[zSlize - nStep + i] * (
                            inputVector[inpIdx - nStep + i] -
                            lastMedian - interpolationStepSize * i
                            ));
                        if (answer[outIdx - nStep * frameSize + i * frameSize] < 0)
                            answer[outIdx - nStep * frameSize + i * frameSize] = 0;
                    }
                    lastMedian = (filterWindow[low + +high / 2] +
                        (high % 2) * filterWindow[low + high % 2 + high / 2]) / (1 + (high % 2)); // median for this round, store for next.
                    
                    

                    high--; // decrease counter.
                    inpIdx += nStep; // load every nStep entry of inputVector
                    //        filtIdx++; //update counter on position to insert next entry to.
                    outIdx += frameSize * nStep; // next save step.
                    zSlize += nStep;
                }// last loop, right part of vector.
                
                inpIdx -= nStep;
                zSlize -= nStep;
                outIdx -= frameSize * nStep; // next save step.
                nStep = depth - zSlize; // remainding steps.
                answer[outIdx + nStep*frameSize] = (int)(meanVector[idx * depth - 1] * (inputVector[idx * depth - 1] -
                      (filterWindow[low + +high / 2] +
                        (high % 2) * filterWindow[low + high % 2 + high / 2]) / (1 + (high % 2))));

                interpolationStepSize = (filterWindow[low + +high / 2] +
                        (high % 2) * filterWindow[low + high % 2 + high / 2]) / (1 + (high % 2)) -
                        lastMedian; // current - last median value.
                interpolationStepSize /= nStep;
                for (int i = 1; i < nStep; i++)
                {
                    answer[outIdx +i * frameSize] =
                        (int)(meanVector[zSlize + i] * (
                        inputVector[inpIdx + i] -
                        lastMedian - interpolationStepSize * i
                        ));
                    if (answer[outIdx + i * frameSize] < 0)
                        answer[outIdx + i * frameSize] = 0;
                }               
            } // end main check of idx.
        } // End medianKernelInterpolate.





        public static float[] generateTest(int N) // generate test vector to simulate mean frame values.
        {
            float[] test = new float[N];
            for (int i = 0; i < test.Length; i++)
            {
                test[i] = 1;
            }
            return test;
        }

        public static float[] generateTest(int N, int depth) // Generate test vector to easily see if median calculations are correct.
        {
            int[] data = { 604, 810, 789, 753, 653, 717, 619, 736, 571, 688, 720, 804, 906, 663, 674, 883, 740, 754, 658, 599, 869, 689, 616, 746, 625, 676, 765, 639, 689, 713, 674, 659, 697, 822, 693, 638, 759, 750, 706, 720, 728, 672, 530, 543, 675, 797, 666, 916, 615, 725, 537, 588, 625, 715, 757, 786, 608, 650, 687, 774, 767, 627, 606, 685, 637, 777, 696, 664, 626, 813, 530, 776, 796, 770, 857, 594, 617, 708, 672, 654, 624, 660, 693, 604, 721, 623, 598, 678, 638, 821, 799, 490, 698, 718, 683, 770, 698, 625, 754, 766, 792, 698, 578, 731, 688, 670, 609, 676, 933, 709, 918, 631, 795, 703, 698, 703, 990, 685, 611, 693, 649, 764, 623, 591, 636, 663, 776, 790, 694, 714, 662, 898, 650, 745, 779, 586, 737, 745, 863, 686, 735, 757, 805, 748, 582, 642, 669, 775, 616, 640, 663, 768, 754, 724, 636, 642, 773, 728, 798, 623, 786, 677, 710, 758, 645, 769, 612, 623, 588, 674, 738, 696, 712, 807, 672, 693, 748, 720, 715, 784, 839, 662, 742, 698, 711, 769, 803, 786, 683, 645, 651, 652, 712, 721, 646, 801, 749, 802, 724, 582, 635, 869, 780, 906, 835, 601, 598, 650, 707, 750, 687, 626, 594, 834, 934, 747, 720, 925, 777, 722, 737, 540, 604, 816, 766, 844, 715, 837, 490, 703, 612, 752, 643, 838, 818, 744, 787, 714, 772, 670, 740, 745, 675, 723, 604, 546, 760, 759, 668, 722, 771, 777, 824, 590, 695, 719, 621, 691, 648, 762, 654, 785, 788, 702, 722, 641, 646, 699, 713, 794, 833, 826, 636, 731, 796, 549, 759, 733, 593, 655, 643, 757, 722, 793, 564, 533, 796, 647, 676, 666, 728, 697, 655, 732, 820, 738, 700, 772, 625, 725, 616, 782, 668, 567, 536, 697, 788, 682, 593, 873, 769, 571, 744, 728, 612, 722, 592, 675, 760, 732, 694, 710, 689, 733, 717, 686, 704, 614, 694, 710, 857, 708, 665, 725, 680, 630, 629, 741, 686, 723, 947, 795, 732, 638, 700, 741, 707, 810, 823, 714, 713, 805, 739, 577, 723, 788, 692, 792, 604, 805, 805, 688, 719, 742, 830, 584, 560, 588, 731, 703, 682, 730, 568, 674, 587, 750, 707, 732, 893, 812, 658, 600, 635, 766, 613, 646, 737, 745, 700, 620, 740, 750, 756, 597, 788, 853, 521, 683, 743, 592, 751, 661, 754, 805, 710, 658, 814, 733, 488, 793, 744, 671, 849, 771, 729, 619, 654, 663, 780, 667, 949, 741, 706, 570, 679, 634, 836, 585, 652, 873, 818, 770, 749, 662, 653, 711, 713, 737, 877, 726, 860, 642, 700, 905, 635, 813, 866, 638, 668, 583, 669, 621, 766, 807, 614, 720, 693, 655, 719, 666, 662, 612, 658, 640, 798, 645, 718, 508, 681, 711, 618, 625, 681, 698, 675, 832, 925, 568, 679, 663, 870, 761, 661, 640, 704, 765, 794, 676, 733, 697, 648, 696, 710, 758, 523, 875, 591, 652, 635, 793, 542, 622, 583, 687, 620, 586, 625, 734, 673, 823, 662, 713, 702, 708, 665, 592, 822, 776, 742, 753, 804, 696, 702, 568, 808, 574, 748, 840, 820, 809, 700, 737, 615, 731, 740, 852, 863, 828, 702, 799, 707, 544, 774, 560, 759, 680, 723, 564, 685, 592, 707, 809, 654, 694, 694, 650, 709, 629, 688, 685, 598, 669, 771, 753, 694, 816, 775, 657, 517, 623, 801, 732, 670, 737, 697, 550, 657, 794, 826, 680, 842, 530, 793, 755, 952, 789, 835, 789, 783, 676, 708, 856, 491, 744, 646, 683, 748, 559, 648, 654, 798, 681, 691, 757, 641, 778, 617, 695, 838, 652, 699, 721, 715, 849, 788, 703, 558, 675, 792, 647, 773, 725, 784, 619, 470, 696, 844, 551, 734, 744, 719, 711, 551, 685, 637, 731, 840, 593, 672, 784, 677, 627, 730, 824, 659, 686, 591, 758, 635, 727, 697, 691, 699, 661, 687, 642, 625, 690, 667, 551, 906, 766, 642, 787, 695, 667, 690, 613, 933, 601, 559, 552, 784, 702, 652, 627, 639, 729, 804, 719, 756, 739, 695, 777, 575, 682, 598, 614, 708, 820, 720, 630, 755, 735, 828, 659, 756, 570, 713, 772, 543, 718, 731, 851, 683, 529, 800, 726, 788, 726, 656, 606, 753, 824, 643, 704, 751, 639, 597, 776, 709, 509, 841, 597, 685, 706, 762, 707, 814, 519, 813, 820, 726, 618, 633, 726, 680, 655, 518, 522, 607, 799, 951, 924, 865, 682, 667, 752, 783, 690, 637, 665, 754, 589, 711, 593, 696, 769, 764, 666, 688, 546, 884, 799, 531, 846, 731, 796, 706, 702, 800, 1060, 856, 798, 622, 557, 810, 656, 838, 751, 685, 717, 776, 779, 527, 728, 818, 587, 732, 689, 734, 835, 641, 728, 764, 624, 755, 631, 563, 663, 640, 692, 670, 830, 603, 797, 776, 773, 714, 758, 828, 720, 628, 748, 628, 667, 847, 773, 605, 689, 740, 637, 827, 624, 619, 740, 633, 644, 762, 755, 800, 611, 675, 827, 649, 730, 770, 729, 771, 711, 641, 651, 716, 739, 602, 613, 758, 715, 695, 802, 758, 839, 651, 645, 777, 711, 702, 703, 681, 719, 692, 690, 716, 734, 778, 849, 804, 846, 867, 634, 642, 744, 634, 627, 746, 516, 511, 761, 646, 639, 641, 758, 554, 606, 633, 547, 748, 589, 661, 733, 683, 723, 747, 638, 718, 736, 557, 673, 765, 825, 796, 891, 800, 764, 622, 722, 679, 571, 936, 867, 751, 621, 659, 744, 649, 831, 814, 845, 844, 729, 986, 777, 693, 788, 575, 544, 731, 755, 736, 846, 692, 623, 597, 704, 574, 624, 638, 743, 722, 717, 811, 696, 862, 729, 646, 600, 675, 599, 617, 714, 859, 827, 747, 602, 705, 695, 882, 781, 614, 630, 742, 660, 855, 698, 565, 695, 522, 699, 732, 767, 645, 714, 617, 645, 691, 724, 705, 587, 761, 593, 611, 599, 943, 887, 900, 683, 645, 792, 842, 709, 637, 724, 574, 718, 726, 818, 942, 710, 674, 787, 783, 779, 639, 686, 737, 713, 747, 861, 685, 938, 798, 740, 744, 753, 707, 724, 734, 664, 820, 873, 698, 958, 674, 603, 637, 677, 727, 587, 839, 724, 800, 724, 707, 581, 682, 662, 670, 660, 556, 720, 699, 719, 827, 614, 725, 796, 807, 736, 751, 712, 500, 680, 709, 590, 768, 833, 718, 593, 660, 632, 695, 785, 735, 750, 754, 751, 664, 868, 679, 519, 526, 780, 670, 645, 752, 717, 712, 782, 713, 620, 764, 712, 576, 931, 586, 860, 706, 728, 580, 736, 714, 811, 706, 749, 745, 697, 602, 777, 646, 676, 608, 589, 847, 672, 859, 601, 637, 665, 753, 672, 777, 956, 748, 683, 593, 697, 815, 762, 740, 701, 759, 854, 705, 744, 739, 762, 698, 787, 793, 658, 662, 551, 712, 753, 825, 609, 801, 484, 712, 686, 755, 623, 593, 657, 688, 666, 741, 737, 771, 821, 774, 683, 615, 730, 776, 548, 593, 697, 813, 709, 620, 783, 657, 622, 759, 849, 862, 805, 829, 591, 730, 667, 691, 636, 789, 754, 591, 657, 719, 595, 624, 587, 709, 823, 582, 561, 554, 804, 747, 737, 719, 682, 874, 568, 686, 845, 805, 701, 615, 646, 807, 712, 646, 693, 574, 766, 797, 696, 725, 659, 587, 614, 787, 594, 902, 738, 691, 565, 603, 690, 652, 488, 635, 723, 747, 703, 756, 625, 646, 718, 676, 663, 835, 614, 698, 717, 599, 753, 764, 682, 775, 921, 745, 835, 800, 639, 694, 633, 840, 827, 647, 583, 651, 716, 721, 652, 798, 543, 609, 696, 658, 753, 692, 650, 759, 726, 788, 863, 580, 648, 588, 560, 653, 581, 813, 578, 902, 729, 715, 755, 686, 806, 759, 768, 1008, 649, 754, 681, 762, 734, 549, 658, 807, 925, 861, 717, 703, 759, 800, 737, 744, 586, 735, 717, 763, 818, 667, 661, 652, 816, 585, 589, 720, 716, 699, 624, 736, 612, 662, 749, 745, 674, 812, 741, 726, 668, 730, 645, 842, 669, 912, 552, 742, 939, 767, 610, 916, 793, 857, 607, 707, 641, 682, 738, 596, 690, 708, 726, 657, 683, 762, 803, 658, 686, 671, 849, 721, 544, 544, 486, 500, 674, 675, 711, 582, 834, 812, 820, 494, 726, 544, 807, 733, 717, 630, 726, 658, 640, 748, 878, 696, 625, 631, 643, 789, 772, 783, 881, 719, 729, 536, 576, 688, 707, 770, 672, 730, 723, 846, 734, 626, 593, 822, 549, 770, 506, 623, 783, 685, 731, 593, 756, 719, 726, 647, 811, 695, 768, 670, 640, 742, 690, 802, 737, 877, 628, 803, 859, 918, 753, 797, 663, 718, 761, 611, 714, 815, 747, 661, 766, 694, 678, 677, 711, 764, 720, 753, 588, 562, 715, 791, 711, 550, 610, 853, 726, 603, 644, 698, 631, 745, 679, 623, 649, 714, 619, 661, 655, 744, 798, 716, 718, 710, 785, 678, 689, 593, 880, 567, 663, 648, 763, 739, 721, 829, 775, 559, 712, 677, 612, 794, 715, 670, 597, 778, 695, 764, 906, 679, 612, 779, 714, 653, 765, 906, 646, 615, 758, 623, 699, 658, 702, 771, 633, 624, 729, 591, 781, 578, 740, 652, 673, 1075, 632, 695, 654, 718, 831, 632, 688, 697, 726, 708, 773, 660, 779, 619, 614, 748, 726, 785, 854, 706, 719, 660, 874, 711, 671, 573, 755, 530, 615, 654, 761, 756, 662, 709, 629, 520, 657, 598, 644, 790, 636, 829, 769, 667, 831, 611, 733, 897, 723, 685, 792, 585, 680, 721, 677, 625, 729, 697, 747, 616, 637, 548, 784, 807, 546, 592, 753, 720, 723, 856, 595, 763, 684, 850, 840, 704, 716, 511, 741, 626, 822, 775, 572, 762, 576, 907, 732, 655, 771, 730, 738, 695, 805, 857, 744, 681, 719, 634, 764, 766, 494, 667, 679, 774, 935, 681, 553, 680, 707, 790, 721, 833, 529, 752, 695, 589, 658, 795, 733, 680, 675, 658, 660, 756, 904, 671, 644, 744, 725, 789, 561, 840, 589, 657, 663, 698, 780, 650, 630, 722, 682, 801, 609, 719, 863, 697, 659, 792, 521, 669, 724, 758, 620, 841, 722, 679, 740, 752, 654, 781, 741, 735, 608, 659, 696, 683, 786, 622, 714, 541, 575, 589, 815, 688, 896, 770, 666, 649, 806, 801, 555, 722, 641, 633, 724, 731, 613, 652, 752, 603, 818, 841, 683, 636, 765, 716, 597, 991, 792, 652, 708, 619, 747, 815, 556, 615, 688, 689, 813, 665, 689, 630, 721, 606, 814, 741, 579, 728, 563, 761, 737, 872, 622, 647, 627, 592, 807, 713, 581, 816, 533, 563, 648, 678, 874, 519, 774, 735, 732, 599, 738, 764, 855, 787, 761, 761, 839, 647, 760, 763, 664, 692, 743, 605, 728, 844, 793, 848, 672, 819, 677, 761, 905, 539, 573, 907, 746, 645, 857, 738, 543, 658, 729, 776, 631, 724, 745, 763, 534, 747, 734, 714, 756, 818, 648, 641, 662, 670, 685, 885, 562, 535, 650, 605, 656, 689, 747, 771, 598, 513, 722, 947, 619, 604, 759, 692, 729, 816, 783, 619, 913, 808, 775, 776, 635, 692, 824, 769, 793, 772, 777, 636, 786, 722, 676, 673, 569, 625, 684, 678, 567, 637, 586, 679, 955, 846, 767, 638, 717, 758, 726, 832, 766, 838, 742, 774, 755, 709, 682, 656, 695, 835, 626, 787, 632, 842, 679, 473, 518, 744, 666, 669, 684, 679, 755, 697, 623, 682, 764, 670, 666, 659, 673, 611, 799, 690, 871, 755, 771, 604, 789, 617, 679, 845, 755, 756, 720, 644, 633, 577, 588, 635, 778, 676, 613, 643, 701, 787, 608, 632, 709, 563, 794, 894, 664, 697, 679, 797, 744, 612, 767, 640, 858, 700, 709, 696, 726, 732, 861, 770, 665, 845, 626, 651, 677, 756, 698, 773, 558, 863, 725, 639, 602, 626, 640, 632, 690, 776, 664, 675, 791, 927, 783, 634, 656, 870, 709, 804, 718, 631, 862, 575, 640, 598, 699, 683, 598, 695, 762, 685, 718, 719, 752, 645, 650, 619, 732, 762, 731, 661, 857, 730, 716, 622, 774, 832, 758, 791, 854, 617, 701, 692, 861, 697, 648, 883, 822, 664, 681, 609, 761, 741, 696, 822, 654, 808, 730, 655, 798, 595, 839, 791, 749, 712, 708, 692, 678, 606, 813, 590, 940, 668, 485, 774, 783, 594, 586, 733, 699, 876, 611, 676, 665, 711, 610, 490, 656, 778, 581, 817, 704, 656, 652, 633, 713, 792, 701, 520, 712, 950, 691, 754, 764, 765, 750, 842, 827, 836, 644, 539, 893, 842, 701, 748, 755, 575, 636, 485, 745, 786, 757, 671, 679, 741, 760, 826, 648, 697, 892, 910, 711, 665, 662, 758, 644, 686, 589, 785, 804, 727, 576, 663, 631, 806, 700, 735, 521, 666, 803, 687, 617, 720, 645, 818, 779, 528, 650, 734, 611, 876, 537, 766, 850, 873, 733, 679, 791, 722, 742, 687, 814, 724, 489, 739, 838, 785, 694, 625, 680, 748, 669, 836, 648, 762, 752, 524, 653, 574, 693, 735, 718, 779, 568, 701, 820, 727, 676, 512, 860, 761, 757, 837, 671, 635, 700, 573, 786, 652, 844, 598, 586, 754, 568, 687, 724, 790, 808, 772, 669, 663, 613, 813, 639, 705, 867, 694, 621, 661, 701, 619, 644, 713, 703, 826, 626, 693, 627, 671, 647, 607, 802, 740, 725, 702, 870, 761, 672, 764, 738, 957, 677, 700, 706, 627, 717, 804, 536, 693, 706, 838, 672, 628, 702, 659, 751, 618, 728, 759, 640, 769, 719, 737, 681, 654, 835, 836, 836, 884, 708, 798, 760, 860, 700, 760, 721, 653, 789, 618, 758, 776, 639, 707, 860, 693, 689, 718, 839, 787, 619, 689, 711, 681, 613, 702, 801, 612, 799, 690, 683, 711, 663, 695, 706, 722, 538, 661, 495, 758, 805, 715, 679, 736, 616, 639, 730, 766, 657, 839, 665, 631, 814, 730, 679, 585, 750, 683, 824, 626, 657, 782, 743, 735, 687, 731, 667, 587, 700, 703, 708, 666, 563, 689, 741, 670, 617, 664, 740, 878, 595, 789, 648, 630, 673, 704, 774, 586, 522, 726, 889, 686, 533, 765, 569, 779, 809, 660, 745, 685, 717, 670, 792, 747, 907, 675, 812, 704, 786, 860, 807, 696, 679, 700, 687, 703, 702, 736, 736, 594, 601, 609, 607, 659, 788, 636, 820, 772, 881, 781, 793, 801, 630, 813, 706, 562, 790, 632, 682, 720, 739, 575, 819, 649, 808, 689, 722, 618, 576, 670, 606, 746, 827, 801, 776, 702, 646, 785, 534, 584, 583, 736, 767, 608, 724, 593, 883, 694, 807, 630, 724, 607, 789, 642, 600, 626, 826, 645, 789, 701, 551, 826, 845, 611, 946, 610, 613, 642, 617, 734, 688, 657, 645, 585, 692, 751, 732, 729, 576, 723, 750, 761, 713, 703, 585, 545, 666, 638, 760, 577, 634, 736, 711, 720, 729, 613, 664, 683, 600, 702, 701, 667, 753, 600, 627, 683, 642, 604, 670, 661, 709, 752, 779, 942, 506, 619, 672, 566, 652, 731, 763, 626, 797, 710, 682, 807, 748, 729, 639, 824, 687, 776, 661, 585, 741, 755, 612, 682 };
            float[] test = new float[N];
            int count = 0;
            for (int i = 0; i < test.Length; i++)
            {
                test[i] = data[count];
                count++;
                if (count == depth)
                    count = 0;
            }
            return test;
        }
    }
}
