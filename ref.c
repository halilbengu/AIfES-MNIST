/*
  www.aifes.ai
  https://github.com/Fraunhofer-IMS/AIfES_for_Arduino
  Copyright (C) 2020-2021 Fraunhofer Institute for Microelectronic Circuits and Systems. All rights reserved.

  AIfES is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <https://www.gnu.org/licenses/>.

  AIfES XOR training demo
  --------------------

    Versions:
    1.0.0   Initial version

  The sketch shows an example of how a neural network is trained from scratch in AIfES using training data.
  As in the example "0_XOR_Inference", an XOR gate is mapped here using a neural network.
  The 4 different states of an XOR gate are fed in as training data here.
  The network structure is 2-3(Sigmoid)-1(Sigmoid) and Sigmoid is used as activation function.
  In the example, the weights are initialized randomly in a range of values from -2 to +2. The Gotrot initialization was inserted as an alternative and commented out.
  For the training the ADAM Optimizer is used, the SGD Optimizer was commented out.
  The optimizer performs a batch training over 100 epochs.
  The calculation is done in float 32.

  XOR truth table / training data
  Input    Output
  0   0    0
  0   1    1
  1   0    1
  1   1    0
  */

#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <time.h>

#include "aifes.h"

#define INPUTS  2
#define NEURONS 3
#define OUTPUTS 1

//For AIfES Express
#define DATASETS        4
#define FNN_3_LAYERS    3
#define PRINT_INTERVAL  10
uint32_t global_epoch_counter = 0;



void AIfES_demo()
{
    printf("AIfES Demo:\n\n");

    uint32_t i;

    // Tensor for the training data
    // Corresponds to the XOR truth table
    float input_data[] = {0.0f, 0.0f,
                0.0f, 1.0f,
                1.0f, 0.0f,
                1.0f, 1.0f};
    // Two dimensional(2D)array example
    // The "printf" output must then be modified
    /*
    float input_data[4][2] = {
    {0.0f, 0.0f},
    {0.0f, 1.0f},
    {1.0f, 0.0f},
    {1.0f, 1.0f}
    };
    */
    uint16_t input_shape[] = {4, INPUTS};    // Definition of the input shape
    aitensor_t input_tensor = AITENSOR_2D_F32(input_shape, input_data); // Creation of the input AIfES tensor with two dimensions and data type F32 (float32)

     // Tensor for the target data
     // Corresponds to the XOR truth table
     float target_data[] = {0.0f,
               1.0f,
               1.0f,
               0.0f};
    uint16_t target_shape[] = {4, OUTPUTS};     // Definition of the output shape
    aitensor_t target_tensor = AITENSOR_2D_F32(target_shape, target_data); // Assign the target_data array to the tensor. It expects a pointer to the array where the data is stored

    // Tensor for the output data (result after training).
    // Same configuration as for the target tensor
    // Corresponds to the XOR truth table
    float output_data[4];
    uint16_t output_shape[] = {4, OUTPUTS};
    aitensor_t output_tensor = AITENSOR_2D_F32(output_shape, output_data);

    // ---------------------------------- Layer definition ---------------------------------------

    // Input layer
    uint16_t input_layer_shape[] = {1, INPUTS};          // Definition of the input layer shape (Must fit to the input tensor)

    ailayer_input_f32_t   input_layer     = AILAYER_INPUT_F32_A( /*input dimension=*/ 2, /*input shape=*/ input_layer_shape);   // Creation of the AIfES input layer
    ailayer_dense_f32_t   dense_layer_1   = AILAYER_DENSE_F32_A( /*neurons=*/ 3); // Creation of the AIfES hidden dense layer with 3 neurons
    ailayer_sigmoid_f32_t sigmoid_layer_1 = AILAYER_SIGMOID_F32_A(); // Hidden activation function
    ailayer_dense_f32_t   dense_layer_2   = AILAYER_DENSE_F32_A( /*neurons=*/ 1); // Creation of the AIfES output dense layer with 1 neuron
    ailayer_sigmoid_f32_t sigmoid_layer_2 = AILAYER_SIGMOID_F32_A(); // Output activation function

    ailoss_mse_t mse_loss;                          //Loss: mean squared error

    // --------------------------- Define the structure of the model ----------------------------

    aimodel_t model;  // AIfES model
    ailayer_t *x;     // Layer object from AIfES, contains the layers

    // Passing the layers to the AIfES model
    model.input_layer = ailayer_input_f32_default(&input_layer);
    x = ailayer_dense_f32_default(&dense_layer_1, model.input_layer);
    x = ailayer_sigmoid_f32_default(&sigmoid_layer_1, x);
    x = ailayer_dense_f32_default(&dense_layer_2, x);
    x = ailayer_sigmoid_f32_default(&sigmoid_layer_2, x);
    model.output_layer = x;

    // Add the loss to the AIfES model
    model.loss = ailoss_mse_f32_default(&mse_loss, model.output_layer);

    aialgo_compile_model(&model); // Compile the AIfES model

    // ------------------------------- Allocate memory for the parameters of the model ------------------------------
    uint32_t parameter_memory_size = aialgo_sizeof_parameter_memory(&model);
    printf("Required memory for parameter (Weights, Bias, ...):");
    printf("%d",parameter_memory_size);
    printf("Byte\n");

    void *parameter_memory = malloc(parameter_memory_size);

    // Distribute the memory to the trainable parameters of the model
    aialgo_distribute_parameter_memory(&model, parameter_memory, parameter_memory_size);

    // ------------------------------- Initialize the parameters ------------------------------


    // Alternative weight initialisation
    /*
    aimath_f32_default_init_glorot_uniform(&dense_layer_1.weights);
    aimath_f32_default_init_zeros(&dense_layer_1.bias);
    aimath_f32_default_init_glorot_uniform(&dense_layer_3.weights);
    aimath_f32_default_init_zeros(&dense_layer_3.bias);
    */

    // Random weights in the value range from -2 to +2
    // The value range of the weights was chosen large, so that learning success is not always given ;)
    float max = 2.0;
    float min = -2.0;
    aimath_f32_default_tensor_init_uniform(&dense_layer_1.weights,max,min);
    aimath_f32_default_tensor_init_uniform(&dense_layer_1.bias,max,min);
    aimath_f32_default_tensor_init_uniform(&dense_layer_2.weights,max,min);
    aimath_f32_default_tensor_init_uniform(&dense_layer_2.bias,max,min);

    // -------------------------------- Define the optimizer for training ---------------------

    aiopti_t *optimizer; // Object for the optimizer

    // Alternative: SGD Gradient descent optimizer
    /*
    aiopti_sgd_f32_t sgd_opti;
    sgd_opti.learning_rate = 1.0f;
    sgd_opti.momentum = 0.0f;

    optimizer = aiopti_sgd_f32_default(&sgd_opti);
    */

    //ADAM optimizer
    aiopti_adam_f32_t adam_opti;
    adam_opti.learning_rate = 0.1f;
    adam_opti.beta1 = 0.9f;
    adam_opti.beta2 = 0.999f;
    adam_opti.eps = 1e-7;

    // Choose the optimizer
    optimizer = aiopti_adam_f32_default(&adam_opti);

    // -------------------------------- Allocate and schedule the working memory for training ---------

    uint32_t memory_size = aialgo_sizeof_training_memory(&model, optimizer);
    printf("Required memory for the training (Intermediate results, gradients, optimization memory): %d Byte\n", memory_size);

    void *memory_ptr = malloc(memory_size);

    // Schedule the memory over the model
    aialgo_schedule_training_memory(&model, optimizer, memory_ptr, memory_size);

    // Initialize the AIfES model
    aialgo_init_model_for_training(&model, optimizer);

    // --------------------------------- Print the result before training ----------------------------------

    uint32_t input_counter = 0;  // Counter to print the inputs/training data

    // Do the inference before training
    aialgo_inference_model(&model, &input_tensor, &output_tensor);

    printf("\n");
    printf("Before training:\n");
    printf("Results:\n");
    printf("input 1:\tinput 2:\treal output:\tcalculated output:\n");

    for (i = 0; i < 4; i++) {
    printf("%f",input_data[input_counter]);
    //Serial.print(((float* ) input_tensor.data)[i]); //Alternative print for the tensor
    input_counter++;
    printf("\t");
    printf("%f",input_data[input_counter]);
    input_counter++;
    printf("\t");
    printf("%f",target_data[i]);
    printf("\t");
    printf("%f\n",output_data[i]);
    //Serial.println(((float* ) output_tensor.data)[i]); //Alternative print for the tensor
    }

    // ------------------------------------- Run the training ------------------------------------

    float loss;
    uint32_t batch_size = 4; // Configuration tip: ADAM=4   / SGD=1
    uint16_t epochs = 100;   // Configuration tip: ADAM=100 / SGD=550
    uint16_t print_interval = 10;

    printf("\n");
    printf("Start training\n");
    for(i = 0; i < epochs; i++)
    {
    // One epoch of training. Iterates through the whole data once
    aialgo_train_model(&model, &input_tensor, &target_tensor, optimizer, batch_size);

    // Calculate and print loss every print_interval epochs
    if(i % print_interval == 0)
    {
      aialgo_calc_loss_model_f32(&model, &input_tensor, &target_tensor, &loss);
      printf("Epoch: ");
      printf("%d",i);
      printf(" Loss: ");
      printf("%f\n",loss);

    }
    }
    printf("Finished training\n\n");

    // ----------------------------------------- Evaluate the trained model --------------------------

    // Do the inference after training
    aialgo_inference_model(&model, &input_tensor, &output_tensor);


    printf("After training:\n");
    printf("Results:\n");
    printf("input 1:\tinput 2:\treal output:\tcalculated output:\n");

    input_counter = 0;

    for (i = 0; i < 4; i++) {
        printf("%f",input_data[input_counter]);
        //Serial.print(((float* ) input_tensor.data)[i]); //Alternative print for the tensor
        input_counter++;
        printf("\t");
        printf("%f",input_data[input_counter]);
        input_counter++;
        printf("\t");
        printf("%f",target_data[i]);
        printf("\t");
        printf("%f\n",output_data[i]);
        //Serial.println(((float* ) output_tensor.data)[i]); //Alternative print for the tensor
    }

    //How to print the weights example
    //Serial.println(((float *) dense_layer_1.weights.data)[0]);
    //Serial.println(((float *) dense_layer_1.bias.data)[0]);

    if(loss > 0.3f)
    {
        printf("\n");
        printf("WARNING\n");
        printf("The loss is very high\n");
    }

    printf("\n");
    printf("A learning success is not guaranteed\n");
    printf("The weights were initialized randomly\n\n");
    printf("copy the weights in the (3_XOR_Inference_keras.ino) example:\n");
    printf("---------------------------------------------------------------------------------\n\n");


    printf("float weights_data_dense_1[] = {\n");

    for (i = 0; i < INPUTS * NEURONS; i++) {

        if(i == INPUTS * NEURONS - 1)
        {
            printf("%ff\n",((float *) dense_layer_1.weights.data)[i]);
        }
        else
        {
            printf("%ff,\n",((float *) dense_layer_1.weights.data)[i]);
        }

    }
    printf("};\n\n");

    printf("float bias_data_dense_1[] = {\n");

    for (i = 0; i < NEURONS; i++) {

        if(i == NEURONS - 1)
        {
            printf("%ff\n",((float *) dense_layer_1.bias.data)[i]);
        }
        else
        {
            printf("%ff,\n",((float *) dense_layer_1.bias.data)[i]);
        }

    }
    printf("};\n\n");

    printf("-------------------------------\n\n");

    printf("float weights_data_dense_2[] = {\n");

    for (i = 0; i < NEURONS * OUTPUTS; i++) {

        if(i == NEURONS * OUTPUTS - 1)
        {
            printf("%ff\n",((float *) dense_layer_2.weights.data)[i]);
        }
        else
        {
            printf("%ff,\n",((float *) dense_layer_2.weights.data)[i]);
        }

    }
    printf("};\n\n");

    printf("float bias_data_dense_2[] = {\n");

    for (i = 0; i < OUTPUTS; i++) {

        if(i == OUTPUTS - 1)
        {
            printf("%ff\n",((float *) dense_layer_2.bias.data)[i]);
        }
        else
        {
            printf("%ff,\n",((float *) dense_layer_2.bias.data)[i]);
        }

    }
    printf("};\n\n");

    free(parameter_memory);
    free(memory_ptr);
}

// The AIfES Express print function for the loss. It can be customized.
void printLoss(float loss)
{
    global_epoch_counter = global_epoch_counter + 1;
    printf("Epoch: %d / Loss: %f\n",global_epoch_counter * PRINT_INTERVAL, loss);

}

void AIfES_Express_demo()
{

    printf("AIfES-Express Demo:\n\n");

    global_epoch_counter = 0;

    uint32_t i;

    // -------------------------------- describe the feed forward neural network ----------------------------------
    // neurons each layer
    // FNN_structure[0] = input layer with 2 inputs
    // FNN_structure[1] = hidden (dense) layer with 3 neurons
    // FNN_structure[2] = output (dense) layer with 1 output
    uint32_t FNN_structure[FNN_3_LAYERS] = {2,3,1};

    // select the activation functions for the dense layer
    AIFES_E_activations FNN_activations[FNN_3_LAYERS - 1];
    FNN_activations[0] = AIfES_E_sigmoid; // Sigmoid for hidden (dense) layer
    FNN_activations[1] = AIfES_E_sigmoid; // Sigmoid for output (dense) layer

    /* possible activation functions
    AIfES_E_relu
    AIfES_E_sigmoid
    AIfES_E_softmax
    AIfES_E_leaky_relu
    AIfES_E_elu
    AIfES_E_tanh
    AIfES_E_softsign
    AIfES_E_linear
    */

    // AIfES Express function: calculate the number of weights needed
    uint32_t weight_number = AIFES_E_flat_weights_number_fnn_f32(FNN_structure,FNN_3_LAYERS);

    printf("Weights: %d\n",weight_number);

    // FlatWeights array
    //float FlatWeights[weight_number];

    // Alternative weight array
    float *FlatWeights;
    FlatWeights = (float *)malloc(sizeof(float)*weight_number);


    // fill the AIfES Express struct
    AIFES_E_model_parameter_fnn_f32 FNN;
    FNN.layer_count = FNN_3_LAYERS;
    FNN.fnn_structure = FNN_structure;
    FNN.fnn_activations = FNN_activations;
    FNN.flat_weights = FlatWeights;

    // -------------------------------- create the tensors ----------------------------------

    float input_data[4][2] = {
    {0.0f, 0.0f},                                                                       // Input data
    {0.0f, 1.0f},
    {1.0f, 0.0f},
    {1.0f, 1.0f}
    };
    uint16_t input_shape[] = {DATASETS, (uint16_t)FNN_structure[0]};                     // Definition of the input shape
    aitensor_t input_tensor = AITENSOR_2D_F32(input_shape, input_data);                 // Macro for the simple creation of a float32 tensor. Also usable in the normal AIfES version

    float target_data[] = {0.0f, 1.0f, 1.0f, 0.0f};                                     // Target Data
    uint16_t target_shape[] = {DATASETS, (uint16_t)FNN_structure[FNN_3_LAYERS - 1]};     // Definition of the target shape
    aitensor_t target_tensor = AITENSOR_2D_F32(target_shape, target_data);              // Macro for the simple creation of a float32 tensor. Also usable in the normal AIfES version

    float output_data[DATASETS];                                                        // Output data
    uint16_t output_shape[] = {DATASETS, (uint16_t)FNN_structure[FNN_3_LAYERS - 1]};     // Definition of the output shape
    aitensor_t output_tensor = AITENSOR_2D_F32(output_shape, output_data);              // Macro for the simple creation of a float32 tensor. Also usable in the normal AIfES version

    // -------------------------------- init weights settings ----------------------------------

    AIFES_E_init_weights_parameter_fnn_f32  FNN_INIT_WEIGHTS;
    FNN_INIT_WEIGHTS.init_weights_method = AIfES_E_init_uniform;

    /* init methods
        AIfES_E_init_uniform
        AIfES_E_init_glorot_uniform
        AIfES_E_init_no_init        //If starting weights are already available or if you want to continue training
    */

    FNN_INIT_WEIGHTS.min_init_uniform = -2; // only for the AIfES_E_init_uniform
    FNN_INIT_WEIGHTS.max_init_uniform = 2;  // only for the AIfES_E_init_uniform
    // -------------------------------- set training parameter ----------------------------------
    AIFES_E_training_parameter_fnn_f32  FNN_TRAIN;
    FNN_TRAIN.optimizer = AIfES_E_adam;
    /* optimizers
        AIfES_E_adam
        AIfES_E_sgd
    */
    FNN_TRAIN.loss = AIfES_E_mse;
    /* loss
        AIfES_E_mse,
        AIfES_E_crossentropy
    */
    FNN_TRAIN.learn_rate = 0.05f;                           // Learning rate is for all optimizers
    FNN_TRAIN.sgd_momentum = 0.0;                           // Only interesting for SGD
    FNN_TRAIN.batch_size = DATASETS;                        // Here a full batch
    FNN_TRAIN.epochs = 1000;                                // Number of epochs
    FNN_TRAIN.epochs_loss_print_interval = PRINT_INTERVAL;  // Print the loss every x times

    // Your individual print function
    // it must look like this: void YourFunctionName(float x)
    FNN_TRAIN.loss_print_function = printLoss;

    //You can enable early stopping, so that learning is automatically stopped when a learning target is reached
    FNN_TRAIN.early_stopping = AIfES_E_early_stopping_on;
    /* early_stopping
        AIfES_E_early_stopping_off,
        AIfES_E_early_stopping_on
    */
    //Define your target loss
    FNN_TRAIN.early_stopping_target_loss = 0.004;

    printf("\n");
    printf("Start training\n");
    printf("Early stopping at: %f\n",FNN_TRAIN.early_stopping_target_loss);

    // -------------------------------- do the training ----------------------------------
    // In the training function, the FNN is set up, the weights are initialized and the training is performed.
    AIFES_E_training_fnn_f32(&input_tensor,&target_tensor,&FNN,&FNN_TRAIN,&FNN_INIT_WEIGHTS,&output_tensor);


    // -------------------------------- do the inference ----------------------------------
    // AIfES Express function: do the inference
    AIFES_E_inference_fnn_f32(&input_tensor,&FNN,&output_tensor);

    // -------------------------------- print the results ----------------------------------

    printf("\n");
    printf("After training:\n");
    printf("Results:\n");
    printf("input 1:\tinput 2:\treal output:\tcalculated output:\n");

    for (i = 0; i < DATASETS; i++) {
        printf("%f\t%f\t%f\t%f\n",input_data[i][0],input_data[i][1],target_data[i],output_data[i]);
    }

    printf("\nWeights for: 0_Universal/2_AIfES_Express_XOR_F32/0_AIfES_Express_XOR_F32_Inference/0_AIfES_Express_XOR_F32_Inference.ino\n");
    printf("copy and paste the weights\n\n");
    printf("float FlatWeights[] = {");

    for (i = 0; i < weight_number; i++) {
        if(i == weight_number - 1)
        {
            printf("%ff",FlatWeights[i]);
        }
        else
        {
            printf("%ff,",FlatWeights[i]);
        }
    }
    printf("};\n\n\n");


    free(FlatWeights);
}

int main(int argc, char *argv[])
{

    time_t t;

    //IMPORTANT
    //AIfES requires random weights for training
    srand((unsigned) time(&t));

    printf("rand test: %d\n",rand());

    AIfES_demo();
    //AIfES_Express_demo();

	system("pause");

	return 0;
}