from tensorflow.python.training.session_run_hook import SessionRunHook

class RSessionRunHook(SessionRunHook):
  def __init__(self, r_begin, r_after_create_session, r_before_run, r_after_run, r_end):
    super(SessionRunHook, self).__init__()
    self.r_begin = r_begin
    self.r_after_create_session = r_after_create_session
    self.r_before_run = r_before_run
    self.r_after_run = r_after_run
    self.r_end = r_end
  
  def begin(self):
    """Called once before using the session.
    When called, the default graph is the one that will be launched in the
    session.  The hook can modify the graph by adding new operations to it.
    After the `begin()` call the graph will be finalized and the other callbacks
    can not modify the graph anymore. Second call of `begin()` on the same
    graph, should not change the graph.
    """
    self.r_begin()
  
  def after_create_session(self, session, coord):
    """Called when new TensorFlow session is created.
    This is called to signal the hooks that a new session has been created. This
    has two essential differences with the situation in which `begin` is called:
    * When this is called, the graph is finalized and ops can no longer be added
        to the graph.
    * This method will also be called as a result of recovering a wrapped
        session, not only at the beginning of the overall session.
    Args:
      session: A TensorFlow Session that has been created.
      coord: A Coordinator object which keeps track of all threads.
    """
    self.r_after_create_session(session, coord)

  def before_run(self, run_context):
    """Called before each call to run().
    You can return from this call a `SessionRunArgs` object indicating ops or
    tensors to add to the upcoming `run()` call.  These ops/tensors will be run
    together with the ops/tensors originally passed to the original run() call.
    The run args you return can also contain feeds to be added to the run()
    call.
    The `run_context` argument is a `SessionRunContext` that provides
    information about the upcoming `run()` call: the originally requested
    op/tensors, the TensorFlow Session.
    At this point graph is finalized and you can not add ops.
    Args:
      run_context: A `SessionRunContext` object.
    Returns:
      None or a `SessionRunArgs` object.
    """
    return self.r_before_run(run_context)

  def after_run(self, run_context, run_values):
    """Called after each call to run().
    The `run_values` argument contains results of requested ops/tensors by
    `before_run()`.
    The `run_context` argument is the same one send to `before_run` call.
    `run_context.request_stop()` can be called to stop the iteration.
    If `session.run()` raises any exceptions then `after_run()` is not called.
    Args:
      run_context: A `SessionRunContext` object.
      run_values: A SessionRunValues object.
    """
    self.r_after_run(run_context, run_values)

  def end(self, session):
    """Called at the end of session.
    The `session` argument can be used in case the hook wants to run final ops,
    such as saving a last checkpoint.
    If `session.run()` raises exception other than OutOfRangeError or
    StopIteration then `end()` is not called.
    Note the difference between `end()` and `after_run()` behavior when
    `session.run()` raises OutOfRangeError or StopIteration. In that case
    `end()` is called but `after_run()` is not called.
    Args:
      session: A TensorFlow Session that will be soon closed.
    """
    self.r_end(session)
