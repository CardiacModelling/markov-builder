import os

import myokit
import networkx as nx

import markov_builder.example_models as example_models


def nontrapped_open_inactivated_state_blocker(output_dir, save_model=False):

    mc = example_models.construct_four_state_chain()
    mc.add_states(('d_s_O', 'd_s_I'))
    drug_rates = [('s_O', 'd_s_O', 'drug_on', 'drug_off'), ('s_I', 'd_s_I', 'drug_on', 'drug_off')]

    for r in drug_rates:
        mc.add_both_transitions(*r)

    positive_rate_expr = ('a*exp(b*V)', ('a', 'b'))
    negative_rate_expr = ('a*exp(-b*V)', ('a', 'b'))
    kon_rate_expr = ('k*D', ('k'))
    koff_rate_expr = ('l', ('l'))

    rate_dictionary = dict(zip(['k1', 'k3', 'k2', 'k4', 'drug_on', 'drug_off'],
                               [positive_rate_expr] * 2 + [negative_rate_expr]
                               * 2 + [kon_rate_expr] + [koff_rate_expr]))

    mc.parameterise_rates(rate_dictionary, ['V', 'D'])
    mc.draw_graph(os.path.join(output_dir, "%s_nontrapped_open_inactivated_state_blocker.html" % mc.name))

    myokitmodel = mc.get_myokit_model(drug_binding=True)

    if save_model:
        myokit.save(filename=os.path.join("%s_nontrapped_open_inactivated_state_blocker.mmt" % mc.name),
                    model=myokitmodel)


def nontrapped_open_state_blocker(output_dir, save_model=False):
    mc = example_models.construct_four_state_chain()
    mc.add_state('d_s_O')
    drug_rates = [('s_O', 'd_s_O', 'drug_on', 'drug_off')]

    for r in drug_rates:
        mc.add_both_transitions(*r)

    positive_rate_expr = ('a*exp(b*V)', ('a', 'b'))
    negative_rate_expr = ('a*exp(-b*V)', ('a', 'b'))
    kon_rate_expr = ('k*D', ('k'))
    koff_rate_expr = ('l', ('l'))

    rate_dictionary = dict(zip(['k1', 'k3', 'k2', 'k4', 'drug_on', 'drug_off'],
                               [positive_rate_expr] * 2 + [negative_rate_expr]
                               * 2 + [kon_rate_expr] + [koff_rate_expr]))

    mc.parameterise_rates(rate_dictionary, ['V', 'D'])
    mc.draw_graph(os.path.join(output_dir, "%s_nontrapped_open_state_blocker.html" % mc.name))

    myokitmodel = mc.get_myokit_model(drug_binding=True)

    if save_model:
        myokit.save(filename=os.path.join(output_dir, "%s_nontrapped_open_state_blocker.mmt" % mc.name), model=myokitmodel)


def trapped_open_inactivated_state_blocker(output_dir, save_model=False):

    mc = example_models.construct_four_state_chain()
    mc.mirror_model(prefix='d_')
    drug_rates = [('s_O', 'd_s_O', 'drug_on', 'drug_off'), ('s_I', 'd_s_I', 'drug_on', 'drug_off')]

    for r in drug_rates:
        mc.add_both_transitions(*r)

    positive_rate_expr = ('a*exp(b*V)', ('a', 'b'))
    negative_rate_expr = ('a*exp(-b*V)', ('a', 'b'))
    kon_rate_expr = ('k*D', ('k'))
    koff_rate_expr = ('l', ('l'))

    rate_dictionary = dict(zip(['k1', 'k3', 'k2', 'k4', 'drug_on', 'drug_off'],
                               [positive_rate_expr] * 2 + [negative_rate_expr]
                               * 2 + [kon_rate_expr] + [koff_rate_expr]))

    mc.parameterise_rates(rate_dictionary, ['V', 'D'])
    mc.draw_graph(os.path.join(output_dir, "%s_trapped_open_inactivated_state_blocker.html" % mc.name))

    myokitmodel = mc.get_myokit_model(drug_binding=True)

    if save_model:
        myokit.save(filename=os.path.join(output_dir, "%s_trapped_open_inactivated_state_blocker.mmt") %
                    mc.name, model=myokitmodel)


def trapped_inactivated_state_blocker(output_dir, save_model=False):

    mc = example_models.construct_four_state_chain()
    mc.mirror_model(prefix='d_')
    drug_rates = [('s_I', 'd_s_I', 'drug_on', 'drug_off')]

    for r in drug_rates:
        mc.add_both_transitions(*r)

    positive_rate_expr = ('a*exp(b*V)', ('a', 'b'))
    negative_rate_expr = ('a*exp(-b*V)', ('a', 'b'))
    kon_rate_expr = ('k*D', ('k'))
    koff_rate_expr = ('l', ('l'))

    rate_dictionary = dict(zip(['k1', 'k3', 'k2', 'k4', 'drug_on', 'drug_off'],
                               [positive_rate_expr] * 2 + [negative_rate_expr]
                               * 2 + [kon_rate_expr] + [koff_rate_expr]))

    mc.parameterise_rates(rate_dictionary, ['V', 'D'])
    mc.draw_graph(os.path.join(output_dir, "%s_trapped_inactivated_state_blocker.html" % mc.name), show_parameters=True)

    myokitmodel = mc.get_myokit_model(drug_binding=True)
    print(myokitmodel.code())

    if save_model:
        myokit.save(filename=os.path.join("%s_trapped_inactivated_state_blocker.mmt") % mc.name, model=myokitmodel)


def main():
    output_dir = os.environ.get('MARKOVBUILDER_EXAMPLE_OUTPUT',
                                os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                             "example_output"))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    nontrapped_open_inactivated_state_blocker(output_dir)
    nontrapped_open_state_blocker(output_dir)
    trapped_open_inactivated_state_blocker(output_dir)
    trapped_inactivated_state_blocker(output_dir)


if __name__ == "__main__":
    main()
