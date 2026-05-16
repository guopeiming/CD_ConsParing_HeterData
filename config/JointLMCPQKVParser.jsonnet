local plm = "./data/plms/bert-base-uncased";
local tokenizer = {
    type: "whitespace",
};
local token_indexers = {
    tokens: {
        type: "pretrained_transformer_mismatched_endpoint",
        model_name: plm,
    },
    // token_characters: {
    //     type: "characters",
    //     min_padding_length: 3
    // },
};

{
    numpy_seed: 2023 + std.parseInt(std.extVar("seed")),
    pytorch_seed: 6 + std.parseInt(std.extVar("seed")),
    random_seed: 23 + std.parseInt(std.extVar("seed")),

    dataset_reader: {
        type: "joint_lm_cp_parser",
        tokenizer: tokenizer,
        token_indexers: token_indexers,
    },

    train_data_path: std.extVar("train_data_path"),
    validation_data_path: std.extVar("dev_data_path"),
    test_data_path: std.extVar("test_data_path"),
    evaluate_on_test: true,
    datasets_for_vocab_creation: ["train"],

    model: {
        type: "joint_lm_cp_qkv_share_parser",
        text_field_embedder: {
            type: "joint_lm_cp_plm_qkv_share",
            model_name: plm,
            requires_grad: true,
            share_prompt_len: 10,
            domain_prompt_len: 10,
            task_prompt_len: 10,
            prompt_dim: 1024,
        },
        encoder: {
            type: "partitioned_transformer",
            input_size: 768,
            num_layers: 2,
            d_model: 1024,
        },
        span_extractor: {
            type: "constituency_span",
            input_dim: 1024,
        },
        structure_lm: true,
        lm_loss_weight: 0.1,
        share_loss_weight: 0.1,
        domain_loss_weight: 0.5,
        task_loss_weight: 0.5,
    },
    data_loader: {
        batch_sampler: {
            type: "bucket",
            batch_size: 16,
            sorting_keys: ["tokens"]
        },
        // batches_per_epoch: 1600,
    },
    validation_data_loader: {
        batch_sampler: {
            type: "bucket",
            batch_size: 64,
            sorting_keys: ["tokens"]
        }
    },
    trainer: {
        num_epochs: 500,
        // grad_norm: 100.0,
        patience: 8,
        validation_metric: "+evalb_f1_measure",
        num_gradient_accumulation_steps: 4,
        run_confidence_checks: false,
        optimizer: {
            type: "adamw",
            lr: 5e-5,
            weight_decay: 0.01,
            betas: [0.9, 0.98],
            eps: 1e-9
        },
        // learning_rate_scheduler: {
        //     type: "polynomial_decay",
        //     warmup_steps: 400,
        //     end_learning_rate: 5e-5,
        // },
        learning_rate_scheduler: {
            type: "reduce_on_plateau_with_warmup",
            warmup_steps: 400,
            mode: "max",
            factor: 0.5,
            patience: 2,
            verbose: true,
            threshold_mode: "abs",
            threshold: 0.0001,
            min_lr: 1e-8,
        },
    },
}
