local plm = "./data/plms/bert-large-uncased";
local tokenizer = {
    type: "whitespace",
};
local token_indexers = {
    tokens: {
        type: "pretrained_transformer_mismatched_endpoint",
        model_name: plm,
    },
};

{
    numpy_seed: 2023 + std.parseInt(std.extVar("seed")),
    pytorch_seed: 6 + std.parseInt(std.extVar("seed")),
    random_seed: 23 + std.parseInt(std.extVar("seed")),

    dataset_reader: {
        type: "base_constituency_parser",
        tokenizer: tokenizer,
        token_indexers: token_indexers,
    },

    train_data_path: std.extVar("train_data_path"),
    validation_data_path: std.extVar("dev_data_path"),
    test_data_path: std.extVar("test_data_path"),
    evaluate_on_test: true,
    datasets_for_vocab_creation: ["train"],

    model: {
        type: "from_archive",
        archive_file: "results/base/v2_3e_sub/"
    },
    data_loader: {
        batch_sampler: {
            type: "bucket",
            batch_size: 30,
            sorting_keys: ["tokens"]
        },
        // batches_per_epoch: 3000,
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
        patience: 8,
        validation_metric: "+evalb_f1_measure",
        num_gradient_accumulation_steps: 1,
        run_confidence_checks: false,
        optimizer: {
            type: "adamw",
            lr: 3e-5,
            weight_decay: 0.01,
            betas: [0.9, 0.98],
            eps: 1e-9
        },
        learning_rate_scheduler: {
            type: "reduce_on_plateau_with_warmup",
            warmup_steps: 400,
            mode: "max",
            factor: 0.5,
            patience: 2,
            verbose: true,
            threshold_mode: "abs",
            threshold: 0.0001,
            min_lr: 1e-9,
        },
    },
}
