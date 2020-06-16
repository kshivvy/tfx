# The Tuner TFX Pipeline Component

The Tuner component tunes the hyperparameters for the model.

## Tuner Component and KerasTuner Library

Tuner component makes extensive use of the Python
[KerasTuner](https://github.com/keras-team/keras-tuner) API for tuning
hyperparameters.

Note: KerasTuner library is generic for tuning, not Keras only.

## Component

Tuner takes:

*   tf.Examples used for training and eval.
*   A user provided module file (or module fn) that defines the tuning logic,
    including model definition, hyperparameter search space, objective etc.
*   [Protobuf](https://developers.google.com/protocol-buffers) definition of
    train args and eval args.
*   (Optional) [Protobuf](https://developers.google.com/protocol-buffers)
    definition of tune args.
*   (Optional) transform graph produced by an upstream Transform component.
*   (Optional) A data schema created by a SchemaGen pipeline component and
    optionally altered by the developer.

With the given data, model and objective, Tuner tunes the hyperparameters and
emits the best result.

## Instruction

A user module function `tuner_fn` with the following signature is required for
Tuner:

```python
...
from kerastuner.engine import base_tuner

TunerFnResult = NamedTuple('TunerFnResult', [('tuner', base_tuner.BaseTuner),
                                             ('fit_kwargs', Dict[Text, Any])])

def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
  """Build the tuner using the KerasTuner API.
  Args:
    fn_args: Holds args as name/value pairs.
      - working_dir: working dir for tuning.
      - train_files: List of file paths containing training tf.Example data.
      - eval_files: List of file paths containing eval tf.Example data.
      - train_steps: number of train steps.
      - eval_steps: number of eval steps.
      - schema_path: optional schema of the input data.
      - transform_graph_path: optional transform graph produced by TFT.
  Returns:
    A namedtuple contains the following:
      - tuner: A BaseTuner that will be used for tuning.
      - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                    model , e.g., the training and validation dataset. Required
                    args depend on the above tuner's implementation.
  """
  ...
```

In this function, user defines the model, hyperparameters search space, and
choose the objective and algorithm for tuning. Tuner component takes this module
code as input, tunes the hyperparameters and emits best result.

Trainer can take Tuner's output best hyperparameters as input and utilize it in
its user module code. The pipeline definition looks like this:

```python
...
tuner = Tuner(
    module_file=module_file,  # Contains `tuner_fn`.
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    train_args=trainer_pb2.TrainArgs(num_steps=20),
    eval_args=trainer_pb2.EvalArgs(num_steps=5))

trainer = Trainer(
    module_file=module_file,  # Contains `run_fn`.
    custom_executor_spec=executor_spec.ExecutorClassSpec(GenericExecutor),
    examples=transform.outputs['transformed_examples'],
    transform_graph=transform.outputs['transform_graph'],
    schema=schema_gen.outputs['schema'],
    # This will be passed to `run_fn`.
    hyperparameters=tuner.outputs['best_hyperparameters'],
    train_args=trainer_pb2.TrainArgs(num_steps=100),
    eval_args=trainer_pb2.EvalArgs(num_steps=5))
...
```

Users might not want to tune the hyperparameters in each run. Once the
hyperparameters is decided, users can remove Tuner from pipeline and use
ImporterNode to import a previous Tuner's output to feed to Trainer.

```python
hparams_importer = ImporterNode(
    instance_name='import_hparams',
    # This can be Tuner's output file or manually edited file. The file contains
    # text format of hyperparameters (kerastuner.HyperParameters.get_config())
    source_uri='path/to/best_hyperparameters.txt',
    artifact_type=HyperParameters)

trainer = Trainer(
    ...
    # An alternative is directly use the tuned hyperparameters in Trainer's user
    # module code and set hyperparameters to None here.
    hyperparameters = hparams_importer.outputs['result'])
```

## Cloud Tuning

WIP

## Links

[E2E Example](https://github.com/tensorflow/tfx/blob/master/tfx/examples/iris/iris_pipeline_native_keras.py)

[Proposal](https://github.com/tensorflow/community/blob/master/rfcs/20200420-tfx-tuner-component.md)
