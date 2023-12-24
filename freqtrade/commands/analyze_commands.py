import os
import logging
from typing import Any, Dict

from freqtrade.configuration import setup_utils_configuration
from freqtrade.enums import RunMode
from freqtrade.exceptions import OperationalException

# Configure the logger
logger = logging.getLogger(__name__)

EXPORT_FILE_TYPES = {
    RunMode.BACKTEST: 'signals',
    RunMode.PAPERRUN: 'paper_results',
    RunMode.LIVE: 'live_results',
}

def setup_analyze_configuration(args: Dict[str, Any], method: RunMode) -> Dict[str, Any]:
    """
    Prepare the configuration for the entry/exit reason analysis module
    :param args: Cli args from Arguments()
    :param method: Bot running mode
    :return: Configuration
    """
    config = setup_utils_configuration(args, method)

    if method in EXPORT_FILE_TYPES:
        export_type = EXPORT_FILE_TYPES[method]
        export_filename = args.get('exportfilename')

        if export_filename:
            if not os.path.exists(export_filename):
                logger.error(f"The specified export file '{export_filename}' does not exist.")
                raise OperationalException(
                    f"The specified export file '{export_filename}' does not exist."
                )
            if not os.path.isfile(export_filename):
                logger.error(f"The specified export file '{export_filename}' is not a valid file.")
                raise OperationalException(
                    f"The specified export file '{export_filename}' is not a valid file."
                )

            # Ensure the export file has the expected extension
            expected_extension = f"{export_type}.pkl"
            if not export_filename.endswith(expected_extension):
                logger.error(
                    f"The specified export file '{export_filename}' has an incorrect extension. "
                    f"Expected: '{expected_extension}'."
                )
                raise OperationalException(
                    f"The specified export file '{export_filename}' has an incorrect extension. "
                    f"Expected: '{expected_extension}'."
                )

            config['exportfilename'] = export_filename
        else:
            logger.error('--exportfilename argument not provided.')
            raise OperationalException(f'--exportfilename argument not provided.')

    return config


def start_analysis_entries_exits(args: Dict[str, Any]) -> None:
    """
    Start analysis script
    :param args: Cli args from Arguments()
    :return: None
    """
    from freqtrade.data.entryexitanalysis import process_entry_exit_reasons

    try:
        # Initialize configuration
        config = setup_analyze_configuration(args, RunMode.BACKTEST)

        logger.info('Starting freqtrade in analysis mode')

        process_entry_exit_reasons(config)

        logger.info('Analysis completed successfully')

    except OperationalException as ex:
        logger.error(f'OperationalException during analysis: {ex}')
    except FileNotFoundError as ex:
        logger.error(f'FileNotFoundError during analysis: {ex}')
    except Exception as ex:
        logger.exception('An unexpected error occurred during analysis.')
        raise ex

