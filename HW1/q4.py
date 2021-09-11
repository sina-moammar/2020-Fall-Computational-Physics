from automata import GameOfLife

loaf_initial = "      \n" \
               "  **  \n" \
               " *  * \n" \
               "  * * \n" \
               "   *  \n" \
               "      "
loaf_sample = GameOfLife(6, 6, loaf_initial)
loaf_sample.render(10)
loaf_sample.animate(name='loaf')


beacon_initial = "      \n" \
                 " **   \n" \
                 " *    \n" \
                 "    * \n" \
                 "   ** \n" \
                 "      "
beacon_sample = GameOfLife(6, 6, beacon_initial)
beacon_sample.render(10)
beacon_sample.animate(name='beacon')


glider_initial = "      \n" \
                 "      \n" \
                 "   *  \n" \
                 " * *  \n" \
                 "  **  \n" \
                 "      "
glider_sample = GameOfLife(6, 6, glider_initial)
glider_sample.render(10)
glider_sample.animate(name='glider')


eater_and_glider_initial = "          \n" \
                           "      *   \n" \
                           "     *    \n" \
                           "     ***  \n" \
                           "          \n" \
                           "   **     \n" \
                           "    *     \n" \
                           " ***      \n" \
                           " *        \n" \
                           "          "
eater_and_glider_sample = GameOfLife(10, 10, eater_and_glider_initial)
eater_and_glider_sample.render(10)
eater_and_glider_sample.animate(name='eater_and_glider')
