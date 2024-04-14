/** @file      trainGPT2.c
 *  @brief     Source file for the functions to train GPT 2.
 *  @date      Created on 2024/04/09      
 */

/* Includes ---------------------------------------------------------------------------------- */
#include "trainGPT2.h"
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

static float *prv_pfMallocAndPointParameters(xParameterTensors_t *xParams, size_t *pulParamSizes);

static void prv_vGpt2BuildFromCheckpoint(xGPT2_t *pxModel, char *pstrCheckpointPath);

static void prv_vDataloaderInit(xDataLoader_t *xpLoader, char *pstrFileName, uint8_t ucB, uint8_t ucT);

static void prv_vDataLoaderReset(xDataLoader_t *pxLoader);


/* Private functions ------------------------------------------------------------------------- */

static float *prv_pfMallocAndPointParameters(xParameterTensors_t *xParams, size_t *pulParamSizes) 
{
    size_t ulNumParameters = 0;

    for (size_t ucI = 0; ucI < NUM_PARAMETER_TENSORS; ucI++) 
    {
        ulNumParameters += pulParamSizes[ucI];
    }

    /* Malloc all parameters all at once */
    float *pfParamsMemory = malloc(ulNumParameters * sizeof(float));

    /* Assign all the tensors */
    float **ppPtrs[] = 
    {
        &xParams->pfWte, &xParams->pfWpe, &xParams->pfLn1w, &xParams->pfLn1b, &xParams->pfQkvw, &xParams->pfQkvb,
        &xParams->pfAttprojw, &xParams->pfAttprojb, &xParams->pfLn2w, &xParams->pfLn2b, &xParams->pfFcw, &xParams->pfFcb,
        &xParams->pfFcprojw, &xParams->pfFcprojb, &xParams->pfLnfw, &xParams->pfLnfb
    };

    float *pfParamsMemoryIterator = pfParamsMemory;

    for (size_t ucI = 0; ucI < NUM_PARAMETER_TENSORS; ucI++) 
    {
        *(ppPtrs[ucI]) = pfParamsMemoryIterator;
        pfParamsMemoryIterator += pulParamSizes[ucI];
    }

    return pfParamsMemory;
}

static void prv_vGpt2BuildFromCheckpoint(xGPT2_t *pxModel, char *pstrCheckpointPath) 
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

    /* Allocate space for all the parameters and read them in */
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

    /* Count the number of paramaters */
    for (size_t ucI = 0; ucI < NUM_PARAMETER_TENSORS; ucI++) 
    {
        ulNumParameters += pxModel->aulParamSizes[ucI];
    }
    printf("num_parameters: %zu\n", ulNumParameters);
    pxModel->ulNumParameters = ulNumParameters;

    /* Read in all the parameters from file */
    pxModel->pfParamsMemory = prv_pfMallocAndPointParameters(&pxModel->xParams, pxModel->aulParamSizes);
    fread(pxModel->pfParamsMemory, sizeof(float), ulNumParameters, pxModelFile);
    fclose(pxModelFile);

    /* Other inits */
    pxModel->pfActsMemory = NULL;
    pxModel->pfGradsMemory = NULL;
    pxModel->pfMemoryM = NULL;
    pxModel->pfMemoryV = NULL;
    pxModel->pfGradsActsMemory = NULL;
    pxModel->pulInputs = NULL;
    pxModel->pulTargets = NULL;
    pxModel->ulBatchSize = 0;
    pxModel->ulSeqLen = 0;
    pxModel->fMeanLoss = -1.0f; /* -1.0f will designate no loss */
}
/*---------------------------------------------------------------------------------------------*/

static void prv_vDataloaderInit(xDataLoader_t *xpLoader, char *pstrFileName, uint8_t ucB, uint8_t ucT) 
{
    xpLoader->ucB = ucB;
    xpLoader->ucT = ucT;

    /* Open the input file for reading */
    xpLoader->pxTokensFile = fopen(pstrFileName, "rb");
    if(xpLoader->pxTokensFile ==NULL)
    {
        printf("Error opening tokens file\n");
        exit(1);
    }
    fseek(xpLoader->pxTokensFile, 0, SEEK_END);
    xpLoader->ullFileSize = ftell(xpLoader->pxTokensFile);
    fseek(xpLoader->pxTokensFile, 0, SEEK_SET);

    if (xpLoader->ullFileSize < (ucB * ucT + 1) * sizeof(int)) {
        printf("Error: file size is too small for the batch size and sequence length\n");
        exit(1);
    }
    /* start at the beginning */
    xpLoader->ullCurrentPosition = 0; 

    /* allocate space for B*T + 1 integers to store the inputs and targets */
    xpLoader->pulBatch = malloc((ucB * ucT + 1) * sizeof(int));
    xpLoader->pulInputs = xpLoader->pulBatch;
    xpLoader->pulTargets = xpLoader->pulBatch + 1; // targets are shifted by one
    xpLoader->ulNumBatches = xpLoader->ullFileSize / (ucB * ucT * sizeof(int));
}
/*---------------------------------------------------------------------------------------------*/

static void prv_vDataLoaderReset(xDataLoader_t *pxLoader) 
{
    pxLoader->ullCurrentPosition = 0;
}
/*---------------------------------------------------------------------------------------------*/

static void prv_vDataloaderNextBatch(xDataLoader_t *pxLoader) 
{
    uint8_t ucB = pxLoader->ucB;
    uint8_t ucT = pxLoader->ucT;

    /* if we are at the end of the file, loop back to the beginning */
    if (pxLoader->ullCurrentPosition + (ucB*ucT+1) * sizeof(int) > pxLoader->ullFileSize) 
    {
        pxLoader->ullCurrentPosition = 0;
    }
    /* read the B*T+1 integers from the file into batch */
    fseek(pxLoader->pxTokensFile, pxLoader->ullCurrentPosition, SEEK_SET);
    fread(pxLoader->pulBatch, sizeof(int), ucB*ucT+1, pxLoader->pxTokensFile);
    /* advance the current position by B*T integers */
    pxLoader->ullCurrentPosition += ucB*ucT * sizeof(int);
}


static void prv_vGpt2Forward(xGPT2_t *pxModel, uint32_t *pulInputs, uint32_t *pulTargets, uint8_t ucB, uint8_t ucT)
{
    /* convenience parameters */
    uint16_t usV = pxModel->xConfig.usVocabSize;
    uint8_t ucL = pxModel->xConfig.ucNumLayers;
    uint8_t ucNH = pxModel->xConfig.ucNumHeads;
    uint8_t ucC = pxModel->xConfig.usChannels;
    float *pfResidual;

    /* ensure the model was initialized or error out */
    if (pxModel->pfParamsMemory == NULL) 
    {
        printf("Error: model was not initialized properly.\n");
        exit(1);
    }

    /* allocate space for all the activations if needed (done here, lazily) */
    if(pxModel->pfActsMemory == NULL) 
    {
        /* record the current B,T as well */
        pxModel->ulBatchSize = ucB;
        pxModel->ulSeqLen = ucT;

        pxModel->ulActSizes[0] = ucB * ucT * ucC; // encoded
        pxModel->ulActSizes[1] = ucL * ucB * ucT * ucC; // ln1
        pxModel->ulActSizes[2] = ucL * ucB * ucT;  // ln1_mean
        pxModel->ulActSizes[3] = ucL * ucB * ucT;  // ln1_rstd
        pxModel->ulActSizes[4] = ucL * ucB * ucT * 3*ucC; // qkv
        pxModel->ulActSizes[5] = ucL * ucB * ucT * ucC;  // atty
        pxModel->ulActSizes[6] = ucL * ucB * ucNH * ucT * ucT;  // preatt
        pxModel->ulActSizes[7] = ucL * ucB * ucNH * ucT * ucT;  // att
        pxModel->ulActSizes[8] = ucL * ucB * ucT * ucC; // attproj
        pxModel->ulActSizes[9] = ucL * ucB * ucT * ucC; // residual2
        pxModel->ulActSizes[10] = ucL * ucB * ucT * ucC; // ln2
        pxModel->ulActSizes[11] = ucL * ucB * ucT; // ln2_mean
        pxModel->ulActSizes[12] = ucL * ucB * ucT; // ln2_rstd
        pxModel->ulActSizes[13] = ucL * ucB * ucT * 4 * ucC; // fch
        pxModel->ulActSizes[14] = ucL * ucB * ucT * 4 * ucC; // fch_gelu
        pxModel->ulActSizes[15] = ucL * ucB * ucT * ucC; // fcproj
        pxModel->ulActSizes[16] = ucL * ucB * ucT * ucC; // residual3
        pxModel->ulActSizes[17] = ucB * ucT * ucC; // lnf
        pxModel->ulActSizes[18] = ucB * ucT; // lnf_mean
        pxModel->ulActSizes[19] = ucB * ucT; // lnf_rstd
        pxModel->ulActSizes[20] = ucB * ucT * usV; // logits
        pxModel->ulActSizes[21] = ucB * ucT * usV; // probs
        pxModel->ulActSizes[22] = ucB * ucT; // losses

        size_t ulNumActivations = 0;
        for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) 
        {
            ulNumActivations += pxModel->ulActSizes[i];
        }

        printf("num_activations: %zu\n", ulNumActivations);
        pxModel->ulNumActivations = ulNumActivations;
        pxModel->pfActsMemory = malloc_and_point_activations(&pxModel->xActs, pxModel->ulActSizes);
        /* also create memory for caching inputs and targets */
        pxModel->pulInputs = malloc(ucB * ucT * sizeof(int));
        /* might be unused if we never have targets but it's small */
        pxModel->pulTargets = malloc(ucB * ucT * sizeof(int)); 
    }
    else
    {
        /* validate B,T is no larger than what was previously allocated */
        /* in principle, we could re-allocate a larger chunk of memory, for now we just error out */
        if (ucB > pxModel->ulBatchSize || ucT > pxModel->ulSeqLen) 
        {
            printf("Error: batch size or sequence length is inadequately large\n");
            printf("Model: B=%d T=%d, Desired: B=%d T=%d\n", pxModel->ulBatchSize, pxModel->ulSeqLen, ucB, ucT);
            exit(1);
        }
    }


    /* cache the inputs/targets */
    memcpy(pxModel->pulInputs, pulInputs, ucB * ucT * sizeof(int));
    if (pulTargets != NULL) 
    {
        memcpy(pxModel->pulTargets, pulTargets, ucB * ucT * sizeof(int));
    }

    /* forward pass */
    xParameterTensors_t xParams = pxModel->xParams; /* for brevity */
    xActivationTensors_t xActs = pxModel->xActs;
    encoder_forward(xActs.pfEncoded, pulInputs, xParams.pfWte, xParams.pfWpe, ucB, ucT, ucC); /* encoding goes into residual[0] */

    for (uint8_t ucl = 0; ucl < ucL; ucl++) 
    {
        pfResidual = ucl == 0 ? xActs.pfEncoded : xActs.pfResidual3 + (ucl-1) * ucB * ucT * ucC;

        // get the pointers of the weights for this layer
        float *pfLln1w = xParams.pfLn1w + ucl * ucC;
        float *pfLln1b = xParams.pfLn1b + ucl * ucC;
        float *pfLqkvw = xParams.pfQkvw + ucl * 3 * ucC * ucC;
        float *pfLqkvb = xParams.pfQkvb + ucl * 3 * ucC;
        float *pfLattprojw = xParams.pfAttprojw + ucl * ucC * ucC;
        float *pfLattprojb = xParams.pfAttprojb + ucl * ucC;
        float *pfLln2w = xParams.pfLn2w + ucl * ucC;
        float *pfLln2b = xParams.pfLn2b + ucl * ucC;
        float *pfLfcw = xParams.pfFcw + ucl * 4 * ucC * ucC;
        float *pfLfcb = xParams.pfFcb + ucl * 4 * ucC;
        float *pfLfcprojw = xParams.pfFcprojw + ucl * ucC * 4 * ucC;
        float *pfLfcprojb = xParams.pfFcprojb + ucl * ucC;

    }
}




/* Exported functions ------------------------------------------------------------------------ */

int main(void)
{
    /* Build the GPT-2 model from a checkpoint */
    xGPT2_t xModel;
    prv_vGpt2BuildFromCheckpoint(&xModel, "gpt2_124M.bin");

    /* Build the DataLoaders from tokens files, for now use tiny_shakespeare if available, else tiny_stories */
    char *pstrTinyStoriesTrain = "data/TinyStories_train.bin";
    char *pstrTinyStoriesVal = "data/TinyStories_val.bin";
    char *pstrTinyShakespeareTrain = "data/tiny_shakespeare_train.bin";
    char *pstrTinyShakespeareVal = "data/tiny_shakespeare_val.bin";
    char *pstrTrainTokens = access(pstrTinyShakespeareTrain, F_OK) != -1 ? pstrTinyShakespeareTrain : pstrTinyStoriesTrain;
    char *pstrTalTokens = access(pstrTinyShakespeareVal, F_OK) != -1 ? pstrTinyShakespeareVal : pstrTinyStoriesVal;
    uint8_t ucB = 4;
    uint8_t ucT = 64;
    xDataLoader_t xTrainLoader;
    xDataLoader_t xValLoader;
    uint8_t ucValNumBatches = 10;
    uint64_t ullRngState = 1337;
    const uint8_t ucGenMaxLength = 64; /* during inference step we'll generate sequences of this many tokens */
    uint32_t ulGenTokens[ucGenMaxLength];
    struct timespec xStart, xEnd;

    prv_vDataloaderInit(&xTrainLoader, pstrTrainTokens, ucB, ucT);
    printf("train dataset num_batches: %d\n", xTrainLoader.ulNumBatches);

    prv_vDataloaderInit(&xValLoader, pstrTalTokens, ucB, ucT);
    printf("val dataset num_batches: %d\n", xValLoader.ulNumBatches);

    /* some memory for generating samples from the model */

    /* train */
    for (uint8_t ucStep = 0; ucStep <= 20; ucStep++) 
    {
        /* once in a while estimate the validation loss */
        if (ucStep % 10 == 0) 
        {
            float fValLoss = 0.0f;
            prv_vDataLoaderReset(&xValLoader);
            for (uint8_t ucI = 0; ucI < ucValNumBatches; ucI++) 
            {
                prv_vDataloaderNextBatch(&xValLoader);


            
            }


        }
    }




    return 0;
}
/*---------------------------------------------------------------------------------------------*/