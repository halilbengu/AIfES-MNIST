#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "aifes.h"
#include "MNIST_test_data.h"
#include "MNIST_test_data_label.h"
#include "MNIST_training_data.h"
#include "MNIST_training_data_label.h"

void eval_test_data(aimodel_t*, aitensor_t*);
void calc_success_rate(aimodel_t*, aitensor_t*);

void AIfES_mnist()
{
    printf("----- AIfES MNIST Training ----- \n");

    // ---------------------------------- Input Sizes ---------------------------------------
    printf("Training Data Size: %lu\n", sizeof(MNIST_training_data) / sizeof(MNIST_training_data[0]) / (28*28));
    printf("Test Data Size for: %lu\n", sizeof(MNIST_test_data) / sizeof(MNIST_test_data[0]) / (28*28));
    printf("Batch Size: %d\n", BATCH_SIZE);

    uint32_t i;
    uint16_t input_shape[] = {1000, 1, 28, 28}; // [sample count, channels, height, width]
    aitensor_t input_tensor = AITENSOR_4D_F32(input_shape, MNIST_training_data);
    uint16_t target_shape[] = {1000, 10};
	aitensor_t target_tensor = AITENSOR_2D_F32(target_shape, MNIST_training_data_label);

    uint16_t test_shape[] = {100, 1, 28, 28}; // [sample count, channels, height, width]
	aitensor_t test_tensor = AITENSOR_4D_F32(test_shape, MNIST_test_data);
    uint16_t test_target_shape[] = {100, 10};
	aitensor_t test_target_tensor = AITENSOR_2D_F32(test_target_shape, MNIST_test_data_label);

	// ---------------------------------- Layers ---------------------------------------
	uint16_t input_layer_shape[]            = {BATCH_SIZE, 1, 28, 28}; // [batch-size, channels, height, width] <- channels first format
	ailayer_input_f32_t input_layer         = AILAYER_INPUT_F32_A(4, input_layer_shape);
	ailayer_conv2d_t conv2d_layer_1         = AILAYER_CONV2D_F32_A(
                                                                     /* filters =*/     16,
                                                                     /* kernel_size =*/ HW(5, 5),
                                                                     /* stride =*/      HW(1, 1),
                                                                     /* dilation =*/    HW(1, 1),
                                                                     /* padding =*/     HW(2, 2)
                                                                    );
	// ailayer_batch_norm_f32_t bn_layer_1     = AILAYER_BATCH_NORM_F32_A(/* momentum =*/ 0.9f, /* eps =*/ 1e-6f);
	ailayer_relu_f32_t relu_layer_1         = AILAYER_RELU_F32_A();
	ailayer_maxpool2d_t maxpool2d_layer_1   = AILAYER_MAXPOOL2D_F32_A(
                                                                     /* pool_size =*/   HW(2, 2),
                                                                     /* stride =*/      HW(2, 1),
                                                                     /* padding =*/     HW(0, 0)
                                                                    );
	ailayer_conv2d_t conv2d_layer_2         = AILAYER_CONV2D_F32_A(
                                                                     /* filters =*/     32,
                                                                     /* kernel_size =*/ HW(5, 5),
                                                                     /* stride =*/      HW(1, 1),
                                                                     /* dilation =*/    HW(1, 1),
                                                                     /* padding =*/     HW(2, 2)
                                                                    );
	// ailayer_batch_norm_f32_t bn_layer_2     = AILAYER_BATCH_NORM_F32_A(/* momentum =*/ 0.9f, /* eps =*/ 1e-6f);
	ailayer_relu_f32_t relu_layer_2         = AILAYER_RELU_F32_A();
    ailayer_maxpool2d_t maxpool2d_layer_2   = AILAYER_MAXPOOL2D_F32_A(
                                                                    /* pool_size =*/   HW(2, 2),
                                                                    /* stride =*/      HW(2, 1),
                                                                    /* padding =*/     HW(0, 0)
                                                                    );
	ailayer_flatten_t flatten_layer         = AILAYER_FLATTEN_F32_A();
	ailayer_dense_f32_t dense_layer_1       = AILAYER_DENSE_F32_A(/* neurons =*/ 10);
	ailayer_softmax_f32_t softmax_layer_3   = AILAYER_SOFTMAX_F32_A();

	ailoss_crossentropy_f32_t ce_loss;

	// --------------------------- Structure of the model ----------------------------
	aimodel_t model;
	ailayer_t *x;

    // // The channels first related functions ("chw" or "cfirst") are used, because the input data is given as channels first format.
    model.input_layer = ailayer_input_f32_default(&input_layer);
    x = ailayer_conv2d_chw_f32_default(&conv2d_layer_1, model.input_layer);
    // x = ailayer_batch_norm_cfirst_f32_default(&bn_layer_1, x);
    x = ailayer_relu_f32_default(&relu_layer_1, x);
    x = ailayer_maxpool2d_chw_f32_default(&maxpool2d_layer_1, x);
    x = ailayer_conv2d_chw_f32_default(&conv2d_layer_2, x);
    // x = ailayer_batch_norm_cfirst_f32_default(&bn_layer_2, x);
    x = ailayer_relu_f32_default(&relu_layer_2, x);
    x = ailayer_maxpool2d_chw_f32_default(&maxpool2d_layer_2, x);
    x = ailayer_flatten_f32_default(&flatten_layer, x);
    x = ailayer_dense_f32_default(&dense_layer_1, x);
    x = ailayer_softmax_f32_default(&softmax_layer_3, x);
    model.output_layer = x;

    model.loss = ailoss_crossentropy_f32_default(&ce_loss, model.output_layer);

    aialgo_compile_model(&model);

	// ------------------------------- Allocate memory for the parameters of the model ------------------------------
	uint32_t parameter_memory_size = aialgo_sizeof_parameter_memory(&model);
	printf("----- Required memory for parameter (Weights, Bias, ...): -----"); 
    printf("%d", parameter_memory_size); 
    printf(" Byte\n");
	void *parameter_memory = malloc(parameter_memory_size);

	// Distribute the memory to the trainable parameters of the model
	aialgo_distribute_parameter_memory(&model, parameter_memory, parameter_memory_size);

	// ------------------------------- Initialize the parameters and optimizer ------------------------------
    srand(33);
	aialgo_initialize_parameters_model(&model);
    aiopti_t *optimizer;

	aiopti_adam_f32_t adam_opti = AIOPTI_ADAM_F32(0.001f, 0.9f, 0.999f, 1e-8f);
	optimizer = aiopti_adam_f32_default(&adam_opti);

	// -------------------------------- Allocate and schedule the working memory for training ---------
	uint32_t working_memory_size = aialgo_sizeof_training_memory(&model, optimizer);
	printf("----- Required memory for training (Intermediate result, gradients, momentums): ----- "); 
    printf("%d", working_memory_size); 
    printf(" Byte\n");
	void *working_memory = malloc(working_memory_size);

	// Schedule the memory over the model
	aialgo_schedule_training_memory(&model, optimizer, working_memory, working_memory_size);

	aialgo_init_model_for_training(&model, optimizer);

    // --------------------- See test results before training ---------------------
    printf("\n----- Evaluate test data before training: ----- \n");
	eval_test_data((aimodel_t*) &model,(aitensor_t*) &test_tensor);

	// ------------------------------------- Run the training ------------------------------------
	float loss;

    printf("----- Start training ----- \n");
    for(i = 0; i < EPOCHS; i++)
    {
        // One epoch of training. Iterates through the whole data once
        aialgo_train_model(&model, &input_tensor, &target_tensor, optimizer, BATCH_SIZE);

        aialgo_calc_loss_model_f32(&model, &input_tensor, &target_tensor, &loss);
        printf("Epoch %d: loss: %.6f\n", i, loss);
    }

	// ----------------------------------------- Evaluate the trained model --------------------------
	printf("\nResults after training:\n");
	eval_test_data(&model, &test_tensor);
    calc_success_rate(&model, &test_tensor);

	free(working_memory);
	free(parameter_memory);
	return;
}

void eval_test_data(aimodel_t *model, aitensor_t *test_tensor){
    int test_size = sizeof(MNIST_test_data) / sizeof(MNIST_test_data[0]) / (28*28);
    float output_data[test_size*10];
	uint16_t output_shape[2] = {test_size, 10};
	aitensor_t output_tensor = AITENSOR_2D_F32(output_shape, output_data);

	aialgo_inference_model(model, test_tensor, &output_tensor);

    int r = (rand() % (test_size - 5))*10; // Pick a random number up to test size, multiplied with label size
    printf("\n----- Test data, Predicted Values | Target Values -----\n");
    for (int i=r; i < r+10; i++){
        printf("\t%f\t%f |", output_data[i],MNIST_test_data_label[i]);
        printf("\t%f\t%f |", output_data[i+10],MNIST_test_data_label[i+10]);
        printf("\t%f\t%f |", output_data[i+20],MNIST_test_data_label[i+20]);
        printf("\t%f\t%f |", output_data[i+30],MNIST_test_data_label[i+30]);
        printf("\t%f\t%f |", output_data[i+40],MNIST_test_data_label[i+40]);
        printf("\n");
    }
}

void calc_success_rate(aimodel_t *model, aitensor_t *test_tensor){
    int test_size = sizeof(MNIST_test_data) / sizeof(MNIST_test_data[0]) / (28*28);
    float output_data[test_size*10];
	uint16_t output_shape[2] = {test_size, 10};
	aitensor_t output_tensor = AITENSOR_2D_F32(output_shape, output_data);

	aialgo_inference_model(model, test_tensor, &output_tensor);

    int r = (rand() % (test_size - 5))*10; // Pick a random number up to test size, multiplied with label size
    int predicted_index, target_index, success=0;
    printf("\n----- Test data, Predicted Values | Target Values -----\n");
    for (int i=0; i < test_size; i++){
        predicted_index=0;
        target_index=0;

        for (int j=0; j<10;j++){
            if (output_data[predicted_index] < output_data[j]){
                    predicted_index=j;
            }
        }
        for (int j=0; j<10;j++){
            if (MNIST_test_data_label[target_index] < MNIST_test_data_label[j]){
                    target_index=j;
            }
        }
        if (target_index == predicted_index) success++;
    }
    float success_rate = success / test_size;
    printf("Success rate: %f", success_rate);
}

int main(int argc, char *argv[])
{
    time_t t;
    srand((unsigned) time(&t));
    printf("rand test: %d\n",rand());

    AIfES_mnist();
	return 0;
}