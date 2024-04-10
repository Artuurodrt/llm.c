/** @file      trainGPT2.c
 *  @brief     Source file for the functions to train GPT 2.
 *  @date      Created on 2024/04/09      
 */

/* Includes ---------------------------------------------------------------------------------- */
#include "trainGPT2.h"
#include <stdio.h>
#include <stdlib.h>
/* Private define ---------------------------------------------------------------------------- */
#define MODEL_HEADER_SIZE               (256)
#define MAX_SEQ_LEN_INDEX               (2)
#define VOCAB_SIZE_INDEX                (3)
#define NUM_LAYERS_INDEX                (4)
#define NUM_HEADS_INDEX                 (5)
#define CHANNELS_INDEX                  (6)
/* Private typedef --------------------------------------------------------------------------- */
/* Private macro ----------------------------------------------------------------------------- */
/* Private enum ------------------------------------------------------------------------------ */
/* Private struct ---------------------------------------------------------------------------- */
/* Private variables ------------------------------------------------------------------------- */
/* Private function prototypes --------------------------------------------------------------- */
/* Private functions ------------------------------------------------------------------------- */
/* Exported functions ------------------------------------------------------------------------ */

void vGpt2BuildFromCheckpoint(xGPT2_t *pxModel, char *pstrCheckpointPath) 
{
    /* Read in model from a checkpoint file */
    FILE *pxModelFile = fopen(pstrCheckpointPath, "rb");
    int32_t slModelHeader[MODEL_HEADER_SIZE] = { 0 };
    int32_t slMaxT = 0;
    int32_t slV = 0;
    int32_t slL = 0;
    int32_t slNH = 0;
    int32_t slC = 0;
    size_t ulNumParameters = 0;

    if (!pxModelFile) 
    { 
        printf("Error opening model file\n"); 
        exit(1); 
    }

    fread(slModelHeader, sizeof(int), MODEL_HEADER_SIZE, pxModelFile);
    if (slModelHeader[0] != 20240326) 
    { 
        printf("Bad magic model file"); 
        exit(1); 
    }
    if (slModelHeader[1] != 1) 
    { 
        printf("Bad version in model file"); 
        exit(1); 
    }

    /* Read in hyperparameters */
    pxModel->xConfig.usMaxSeqLen = (uint8_t)slModelHeader[MAX_SEQ_LEN_INDEX];
    slMaxT = slModelHeader[MAX_SEQ_LEN_INDEX];

    pxModel->xConfig.usVocabSize = (uint16_t)slModelHeader[VOCAB_SIZE_INDEX];
    slV = slModelHeader[VOCAB_SIZE_INDEX];

    pxModel->xConfig.ucNumLayers = (uint8_t)slModelHeader[NUM_LAYERS_INDEX];
    slL = slModelHeader[NUM_LAYERS_INDEX];

    pxModel->xConfig.ucNumHeads = (uint8_t)slModelHeader[NUM_HEADS_INDEX];
    slNH = slModelHeader[NUM_HEADS_INDEX];

    pxModel->xConfig.usChannels = (uint16_t)slModelHeader[CHANNELS_INDEX];
    slC = slModelHeader[CHANNELS_INDEX];

    printf("[GPT-2]\n");
    printf("max_seq_len: %u\n", slMaxT);
    printf("vocab_size: %u\n", slV);
    printf("num_layers: %u\n", slL);
    printf("num_heads: %u\n", slNH);
    printf("channels: %u\n", slC);

    /* allocate space for all the parameters and read them in */
    pxModel->aulParamSizes[0] = slV * slC;
    pxModel->aulParamSizes[1] = slMaxT * slC;
    pxModel->aulParamSizes[2] = slL * slC;
    pxModel->aulParamSizes[3] = slL * slC;
    pxModel->aulParamSizes[4] = slL * (3 * slC) * slC;
    pxModel->aulParamSizes[5] = slL * (3 * slC);
    pxModel->aulParamSizes[6] = slL * slC * slC;
    pxModel->aulParamSizes[7] = slL * slC;
    pxModel->aulParamSizes[8] = slL * slC;
    pxModel->aulParamSizes[9] = slL * slC;
    pxModel->aulParamSizes[10] = slL * (4 * slC) * slC;
    pxModel->aulParamSizes[11] = slL * (4 * slC);
    pxModel->aulParamSizes[12] = slL * slC * (4 * slC);
    pxModel->aulParamSizes[13] = slL * slC;
    pxModel->aulParamSizes[14] = slC;
    pxModel->aulParamSizes[15] = slC;

    /* count the number of paramaters */
    for (size_t ucI = 0; ucI < NUM_PARAMETER_TENSORS; ucI++) 
    {
        ulNumParameters += pxModel->aulParamSizes[ucI];
    }
    printf("num_parameters: %zu\n", ulNumParameters);


}
/*---------------------------------------------------------------------------------------------*/


int main(void)
{
    /* build the GPT-2 model from a checkpoint */
    xGPT2_t xModel;
    vGpt2BuildFromCheckpoint(&xModel, "gpt2_124M.bin");


    return 0;
}
/*---------------------------------------------------------------------------------------------*/