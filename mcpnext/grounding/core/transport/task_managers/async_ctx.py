"""
Generic connection manager based on an *async context manager*.
Give it any factory that returns an async–context-manager.  
"""
import sys
from typing import Any, Callable, Generic, Optional, ParamSpec, TypeVar
from .base import BaseConnectionManager

# BaseExceptionGroup only exists in Python 3.11+
if sys.version_info >= (3, 11):
    _BaseExceptionGroup = BaseExceptionGroup
else:
    # Dummy class for older Python versions
    class _BaseExceptionGroup(Exception):
        pass

T = TypeVar("T")                # Return type of the async context
P = ParamSpec("P")              # Parameter specification of the factory


class AsyncContextConnectionManager(Generic[T, P], BaseConnectionManager[T]):
    def __init__(self,
                 ctx_factory: Callable[P, Any],
                 *args: P.args,
                 **kwargs: P.kwargs):
        super().__init__()
        self._factory = ctx_factory
        self._factory_args = args
        self._factory_kwargs = kwargs
        self._ctx: Optional[Any] = None 

    async def _establish_connection(self) -> T:
        """Create the context manager and enter it."""
        self._logger.debug("Creating context via %s", self._factory.__name__)
        self._ctx = self._factory(*self._factory_args, **self._factory_kwargs)
        result: T = await self._ctx.__aenter__()
        self._logger.debug("Context %s entered successfully", self._factory.__name__)
        return result

    async def _close_connection(self) -> None:
        """Exit the context manager if it exists.
        
        Uses try-finally to ensure ctx is cleared even if __aexit__ fails.
        This prevents resource leaks when cleanup encounters errors.
        """
        if self._ctx is not None:
            try:
                self._logger.debug("Exiting context %s", self._factory.__name__)
                
                # Give subprocesses a moment to flush buffers before closing
                import asyncio
                await asyncio.sleep(0.05)
                
                # Try to exit the context, but catch all possible exceptions
                try:
                    await self._ctx.__aexit__(None, None, None)
                except BaseException as e:
                    # Catch absolutely everything including SystemExit, KeyboardInterrupt, etc.
                    # Check if it's a benign error
                    benign_error_types = (
                        BrokenPipeError, ConnectionResetError, ValueError, 
                        OSError, IOError, ProcessLookupError, RuntimeError,
                        GeneratorExit
                    )
                    
                    is_benign = False
                    
                    # Check direct exception type
                    if isinstance(e, benign_error_types):
                        is_benign = True
                    # Check for BaseExceptionGroup (Python 3.11+)
                    elif hasattr(e, 'exceptions'):
                        # It's an exception group, check all sub-exceptions
                        is_benign = all(isinstance(sub_e, benign_error_types) for sub_e in e.exceptions)
                    
                    if is_benign:
                        self._logger.debug(f"Benign cleanup error for {self._factory.__name__}: {type(e).__name__}")
                    else:
                        self._logger.warning(f"Error during context exit for {self._factory.__name__}: {type(e).__name__}: {e}")
                    
                    # Don't re-raise - we want cleanup to complete
                    
            except Exception as e:
                # Catch any other unexpected errors in the outer try block
                self._logger.warning(f"Unexpected error during cleanup for {self._factory.__name__}: {e}")
            finally:
                self._ctx = None   