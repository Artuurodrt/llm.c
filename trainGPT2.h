/** @file      trainGPT2.h
 *  @brief     Header file for the functions to train GPT 2.
 *  @date      Created on 2024/04/09   
 */

/* Header guard ------------------------------------------------------------------------------ */
#ifndef TRAINGPT2_H_
#define TRAINGPT2_H_
/* Includes ---------------------------------------------------------------------------------- */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>

/* Define ------------------------------------------------------------------------------------ */
#define NUM_PARAMETER_TENSORS               (16)
#define NUM_ACTIVATION_TENSORS              (23)

/* Exported Macro ---------------------------------------------------------------------------- */
/* Exported Types ---------------------------------------------------------------------------- */
/* Exported Enums ---------------------------------------------------------------------------- */
/* Exported struct --------------------------------------------------------------------------- */
/**
 * @brief Structure representing configuration parameters for a GPT-2 model.
 */
typedef struct 
{
    uint16_t usMaxSeqLen; /* Max sequence length, e.g. 1024 */
    uint16_t usVocabSize; /* Vocab size, e.g. 50257 */
    uint8_t ucNumLayers;  /* Number of layers, e.g. 12 */
    uint8_t ucNumHeads;   /* Number of heads in attention, e.g. 12 */
    uint16_t usChannels;  /* Number of channels, e.g. 768 */
} 
xGPT2Config_t;

/**
 * @brief Structure representing the parameters of the model.
 */
typedef struct 
{
    float *pfWte;      /* (V, C) */
    float *pfWpe;      /* (maxT, C) */
    float *pfLn1w;     /* (L, C) */
    float *pfLn1b;     /* (L, C) */
    float *pfQkvw;     /* (L, 3*C, C) */
    float *pfQkvb;     /* (L, 3*C) */
    float *pfAttprojw; /* (L, C, C) */
    float *pfAttprojb; /* (L, C) */
    float *pfLn2w;     /* (L, C) */
    float *pfLn2b;     /* (L, C) */
    float *pfFcw;      /* (L, 4*C, C) */
    float *pfFcb;      /* (L, 4*C) */
    float *pfFcprojw;  /* (L, C, 4*C) */
    float *pfFcprojb;  /* (L, C) */
    float *pfLnfw;     /* (C) */
    float *pfLnfb;     /* (C) */
}
xParameterTensors_t;

/**
 * @brief Structure representing activation tensors.
 *
 * This structure holds various activation tensors used in the GPT-2 model.
 * The dimensions of each tensor are specified in the comments.
 */
typedef struct 
{
    float *pfEncoded;   /* (B, T, C) */
    float *pfLn1;       /* (L, B, T, C) */
    float *pfLn1_mean;  /* (L, B, T) */
    float *pfLn1_rstd;  /* (L, B, T) */
    float *pfQkv;       /* (L, B, T, 3*C) */
    float *pfAtty;      /* (L, B, T, C) */
    float *pfPreatt;    /* (L, B, NH, T, T) */
    float *pfAtt;       /* (L, B, NH, T, T) */
    float *pfAttproj;   /* (L, B, T, C) */
    float *pfResidual2; /* (L, B, T, C) */
    float *pfLn2;       /* (L, B, T, C) */
    float *pfLn2_mean;  /* (L, B, T) */
    float *pfLn2_rstd;  /* (L, B, T) */
    float *pfFch;       /* (L, B, T, 4*C) */
    float *pfFch_gelu;  /* (L, B, T, 4*C) */
    float *pfFcproj;    /* (L, B, T, C) */
    float *pfResidual3; /* (L, B, T, C) */
    float *pfLnf;       /* (B, T, C) */
    float *pfLnf_mean;  /* (B, T) */
    float *pfLnf_rstd;  /* (B, T) */
    float *pfLogits;    /* (B, T, V) */
    float *pfProbs;     /* (B, T, V) */
    float *pfLosses;    /* (B, T) */
} 
xActivationTensors_t;

/**
 * @brief Structure representing a GPT-2 model instance.
 *
 * This structure encapsulates the configuration, parameters, gradients, memory buffers,
 * activations, and other run state information of a GPT-2 model.
 */
typedef struct 
{
    xGPT2Config_t xConfig;
    xParameterTensors_t xParams;    /* The weights of the model, and their sizes */
    size_t aulParamSizes[NUM_PARAMETER_TENSORS];
    float *pfParamsMemory;
    uint32_t ulNumParameters;
    xParameterTensors_t xGrads; /* Gradients of the weights */
    float *pfGradsMemory;
    float *pfMemoryM; /* Buffers for the AdamW optimizer */
    float *pfMemoryV;
    xActivationTensors_t xActs; /* The activations of the model, and their sizes */
    size_t ulActSizes[NUM_ACTIVATION_TENSORS];
    float *pfActsMemory;
    uint32_t ulNumActivations; /* Gradients of the activations */
    xActivationTensors_t xGradsActs;
    float *pfGradsActsMemory; /* Other run state configuration */
    uint32_t ulBatchSize; /* The batch size (B) of current forward pass */
    uint32_t ulSeqLen; /* The sequence length (T) of current forward pass */
    uint32_t *pulInputs; /* The input tokens for the current forward pass */
    uint32_t *pulTargets; /* The target tokens for the current forward pass */
    float fMeanLoss; /* After a forward pass with targets, will be populated with the mean loss */
} 
xGPT2_t;

/**
 * @brief Structure for data loading.
 * 
 * This structure defines hyperparameters, input handling and its state,
 * output memory, and convenience variables for data loading.
 */
typedef struct 
{
    /* Hyperparameters */
    uint8_t ucB;
    uint8_t ucT;
    /* input handling and its state */
    FILE *pxTokensFile;
    uint64_t ullFileSize;
    uint64_t ullCurrentPosition;
    /* output memory */
    uint32_t *pulBatch;
    uint32_t *pulInputs;
    uint32_t *pulTargets;
    // convenience variables
    uint32_t ulNumBatches;
} 
xDataLoader_t;


/* Global variables -------------------------------------------------------------------------- */
/* Exported function prototypes -------------------------------------------------------------- */



#endif /* TRAINGPT2_H_ */