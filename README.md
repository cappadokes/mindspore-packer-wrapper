The following changes have been made to the original mindpsore code, in order to facilitate an easier isolation of the packer:
- Usage of MS_LOG has been replaced by std::cout
- Usage of MS_EXCEPTION checks has been replaced by assert calls
- Time measurements are in microseconds, rather than in milliseconds
- Usage of session:KernelGraph and related log functions have been removed from the somas_solver_pre source files