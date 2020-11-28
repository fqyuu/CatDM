import tensorflow as tf
import tensorflow.contrib.rnn as rnn

def sequence_loss(logits,
                  targets,
                  weights,
                  average_across_timesteps=True,
                  average_across_batch=True,
                  softmax_loss_function=None,
                  name=None):
    """Weighted cross-entropy loss for a sequence of logits.

    Depending on the values of `average_across_timesteps` and
    `average_across_batch`, the return Tensor will have rank 0, 1, or 2 as these
    arguments reduce the cross-entropy at each target, which has shape
    `[batch_size, sequence_length]`, over their respective dimensions. For
    example, if `average_across_timesteps` is `True` and `average_across_batch`
    is `False`, then the return Tensor will have shape `[batch_size]`.

    Args:
      logits: A Tensor of shape
        `[batch_size, sequence_length, num_decoder_symbols]` and dtype float.
        The logits correspond to the prediction across all classes at each
        timestep.
      targets: A Tensor of shape `[batch_size, sequence_length]` and dtype
        int. The target represents the true class at each timestep.
      weights: A Tensor of shape `[batch_size, sequence_length]` and dtype
        float. `weights` constitutes the weighting of each prediction in the
        sequence. When using `weights` as masking, set all valid timesteps to 1
        and all padded timesteps to 0, e.g. a mask returned by `tf.sequence_mask`.
      average_across_timesteps: If set, sum the cost across the sequence
        dimension and divide the cost by the total label weight across timesteps.
      average_across_batch: If set, sum the cost across the batch dimension and
        divide the returned cost by the batch size.
      softmax_loss_function: Function (labels, logits) -> loss-batch
        to be used instead of the standard softmax (the default if this is None).
        **Note that to avoid confusion, it is required for the function to accept
        named arguments.**
      name: Optional name for this operation, defaults to "sequence_loss".

    Returns:
      A float Tensor of rank 0, 1, or 2 depending on the
      `average_across_timesteps` and `average_across_batch` arguments. By default,
      it has rank 0 (scalar) and is the weighted average cross-entropy
      (log-perplexity) per symbol.

    Raises:
      ValueError: logits does not have 3 dimensions or targets does not have 2
                  dimensions or weights does not have 2 dimensions.
    """
    if len(logits.get_shape()) != 3:
        raise ValueError("Logits must be a "
                         "[batch_size x sequence_length x logits] tensor")
    if len(targets.get_shape()) != 2:
        raise ValueError("Targets must be a [batch_size x sequence_length] "
                         "tensor")
    if len(weights.get_shape()) != 2:
        raise ValueError("Weights must be a [batch_size x sequence_length] "
                         "tensor")
    with tf.name_scope(name, "sequence_loss", [logits, targets, weights]):
        num_classes = tf.shape(logits)[2]
        logits_flat = tf.reshape(logits, [-1, num_classes])
        targets = tf.reshape(targets, [-1])
        if softmax_loss_function is None:
            # [batch_size * seq_len]
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=targets, logits=logits_flat)
        else:
            crossent = softmax_loss_function(labels=targets, logits=logits_flat)
        # 通过weights做mask
        crossent *= tf.reshape(weights, [-1])
        if average_across_timesteps and average_across_batch:
            crossent = tf.reduce_sum(crossent)
            total_size = tf.reduce_sum(weights)
            total_size += 1e-12  # to avoid division by 0 for all-0 weights
            crossent /= total_size
        else:
            batch_size = tf.shape(logits)[0]
            sequence_length = tf.shape(logits)[1]
            crossent = tf.reshape(crossent, [batch_size, sequence_length])
        if average_across_timesteps and not average_across_batch:
            crossent = tf.reduce_sum(crossent, axis=[1])
            total_size = tf.reduce_sum(weights, axis=[1])
            total_size += 1e-12  # to avoid division by 0 for all-0 weights
            crossent /= total_size
        if not average_across_timesteps and average_across_batch:
            crossent = tf.reduce_sum(crossent, axis=[0])
            total_size = tf.reduce_sum(weights, axis=[0])
            total_size += 1e-12  # to avoid division by 0 for all-0 weights
            crossent /= total_size
        return crossent


def linear(args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.

      Args:
        args: a 2D Tensor or a list of 2D, batch x n, Tensors.
        output_size: int, second dimension of W[i].
        bias: boolean, whether to add a bias term or not.
        bias_start: starting value to initialize the bias; 0 by default.
        scope: VariableScope for the created subgraph; defaults to "Linear".

      Returns:
        A 2D Tensor with shape [batch x output_size] equal to
        sum_i(args[i] * W[i]), where W[i]s are newly created matrices.

      Raises:
        ValueError: if some of the arguments has unspecified or wrong shape.
      """
    if args is None or (isinstance(args, (list, tuple)) and not args):
        raise ValueError('`args` must be specified')
    if not isinstance(args, (list, tuple)):
        args = [args]

        # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError('Linear is expecting 2D arguments: %s' % str(shapes))
        if not shape[1]:
            raise ValueError('Linear expects shape[1] of arguments: %s' % str(shapes))
        else:
            total_arg_size += shape[1]

    # Now the computation.
    with tf.variable_scope(scope or 'Linear'):
        matrix = tf.get_variable('Matrix', [total_arg_size, output_size],
                                 initializer=tf.orthogonal_initializer())

        if len(args) == 1:
            res = tf.matmul(args[0], matrix)
        else:
            res = tf.matmul(tf.concat(axis=1, values=args), matrix)
        if not bias:
            return res
        bias_term = tf.get_variable(
            'Bias', [output_size],
            initializer=tf.constant_initializer(bias_start))
    return res + bias_term
