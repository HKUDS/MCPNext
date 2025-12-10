"""
Base connection manager for all backend connectors.

This module provides an abstract base class for different types of connection
managers used in all backend connectors.

Flow: start() → launch_connection_task() → call subclass _establish_connection() → notify ready → maintain connection until stop() → call subclass _close_connection() → cleanup
"""
import asyncio
from abc import ABC, abstractmethod
from typing import Generic, TypeVar
from mcpnext.utils.logging import Logger

T = TypeVar("T")


class BaseConnectionManager(Generic[T], ABC):
    """Abstract base class for connection managers.

    This class defines the interface for different types of connection managers
    used with all backend connectors.
    """

    def __init__(self):
        """Initialize a new connection manager."""
        self._ready_event = asyncio.Event()
        self._done_event = asyncio.Event()
        self._exception: Exception | None = None
        self._connection: T | None = None
        self._task: asyncio.Task | None = None
        self._logger = Logger.get_logger(f"{__name__}.{self.__class__.__name__}")

    @abstractmethod
    async def _establish_connection(self) -> T:
        """Establish the connection.

        This method should be implemented by subclasses to establish
        the specific type of connection needed.

        Returns:
            The established connection.

        Raises:
            Exception: If connection cannot be established.
        """
        pass

    @abstractmethod
    async def _close_connection(self) -> None:
        """Close the connection.

        This method should be implemented by subclasses to close
        the specific type of connection.

        """
        pass

    async def start(self) -> T:
        """Start the connection manager and establish a connection.

        Returns:
            The established connection.

        Raises:
            Exception: If connection cannot be established.
        """
        # Reset state
        self._ready_event.clear()
        self._done_event.clear()
        self._exception = None

        # Create a task to establish and maintain the connection
        self._task = asyncio.create_task(self._connection_task(), name=f"{self.__class__.__name__}_task")

        # Wait for the connection to be ready or fail
        await self._ready_event.wait()

        # If there was an exception, raise it
        if self._exception:
            self._logger.error(f"Failed to start connection: {self._exception}")
            raise self._exception

        # Return the connection
        if self._connection is None:
            error_msg = "Connection was not established"
            self._logger.error(error_msg)
            raise RuntimeError(error_msg)
            
        self._logger.info("Connection manager started successfully")
        return self._connection

    async def stop(self) -> None:
        """Stop the connection manager and close the connection.
        
        Ensures all async resources (including aiohttp sessions) are properly closed.
        """
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass  # Expected
            except Exception as e:
                self._logger.warning(f"Error stopping task: {e}")

        # Wait for the connection to be done
        await self._done_event.wait()
        
        self._logger.info("Connection manager stopped")

    def get_streams(self) -> T | None:
        """Get the current connection streams.

        Returns:
            The current connection (typically a tuple of read_stream, write_stream) or None if not connected.
        """
        return self._connection

    async def _connection_task(self) -> None:
        """Run the connection task.

        This task establishes and maintains the connection until cancelled.
        """
        try:
            # Establish the connection
            self._connection = await self._establish_connection()
            self._logger.debug("Connection established")

            # Signal that the connection is ready
            self._ready_event.set()

            # Wait indefinitely until cancelled
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                raise

        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Store the exception
            self._exception = e
            self._logger.error(f"Connection task failed: {e}")
            # Signal that the connection is ready (with error)
            self._ready_event.set()

        finally:
            # Close the connection if it was established
            if self._connection is not None:
                try:
                    await self._close_connection()
                except Exception as e:
                    self._logger.warning(f"Error closing connection: {e}")
                self._connection = None

            # Signal that the connection is done
            self._done_event.set()