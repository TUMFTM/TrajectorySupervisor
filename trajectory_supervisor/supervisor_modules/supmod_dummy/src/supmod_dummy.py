

class SupModDummy(object):
    """
    Sample supervisor module class, holding the basic layout. In order to set up a new supervisor module, simply copy
    this folder ('`supmod_dummy`'), change all occurrences of 'dummy' to your module name and add your code to the
    function skeleton.

    :Authors:
        * Tim Stahl <tim.stahl@tum.de>

    :Created on:
        07.11.2019

    """

    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------------------------------------------------------
    # DESTRUCTOR -------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def __del__(self) -> None:
        pass

    # ------------------------------------------------------------------------------------------------------------------
    # CALC SCORE -------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    def calc_score(self,
                   dummy_input: float = None) -> bool:
        """
        Calculate safety score (Place description of the function here - stick to the same format).

        :param dummy_input:         this is the description of the dummy_input
        :returns:
            * **safety_score** -    this is the description of the return value

        """
        if dummy_input is not None:
            print(dummy_input)

        return True
