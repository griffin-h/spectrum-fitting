.. _basicexamples:

Advanced Example
========================


More advanced example here.

.. code:: python

	This is python code

.. math::
    :label: eq1

    \nabla^2 f =
    \frac{1}{r^2} \frac{\partial}{\partial r}
    \left( r^2 \frac{\partial f}{\partial r} \right) +
    \frac{1}{r^2 \sin \theta} \frac{\partial f}{\partial \theta}
    \left( \sin \theta \, \frac{\partial f}{\partial \theta} \right) +
    \frac{1}{r^2 \sin^2\theta} \frac{\partial^2 f}{\partial \phi^2}


You can add a link to equations like the one above :eq:`eq1` by using ``:eq:``. Unfortunately the label does not align well with the rtd sphinx theme, this is a known bug...
