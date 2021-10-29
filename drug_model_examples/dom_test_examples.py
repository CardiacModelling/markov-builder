import myokit
import networkx as nx

import markov_builder.example_models as example_models


def nontrapped_open_inactivated_state_blocker(save_file = False):

    mc = example_models.construct_four_state_chain()
    mc.add_states(('d_s_O', 'd_s_I'))
    drug_rates = [('s_O', 'd_s_O', 'drug_on', 'drug_off'), ('s_I', 'd_s_I', 'drug_on', 'drug_off')]

    for r in drug_rates:
        mc.add_both_transitions(*r)

    positive_rate_expr = ('a*exp(b*V)', ('a', 'b'))
    negative_rate_expr = ('a*exp(-b*V)', ('a', 'b'))
    kon_rate_expr = ('k*D', ('k'))
    koff_rate_expr = ('l', ('l'))

    rate_dictionary = dict(zip(['k1', 'k3', 'k2', 'k4', 'drug_on', 'drug_off'], \
        [positive_rate_expr] * 2 + [negative_rate_expr] * 2 + [kon_rate_expr] + [koff_rate_expr]))

    mc.parameterise_rates(rate_dictionary, ['V', 'D'])
    mc.draw_graph("%s_nontrapped_open_inactivated_state_blocker.html" % mc.name)

    myokitmodel = mc.get_myokit_model(drug_binding=True)
    
    if save_file:
        myokit.save(filename="%s_nontrapped_open_inactivated_state_blocker.mmt" % mc.name, model=myokitmodel)

def nontrapped_open_state_blocker(save_file = False):

    mc = example_models.construct_four_state_chain()
    mc.add_state('d_s_O')
    drug_rates = [('s_O', 'd_s_O', 'drug_on', 'drug_off')]

    for r in drug_rates:
        mc.add_both_transitions(*r)

    positive_rate_expr = ('a*exp(b*V)', ('a', 'b'))
    negative_rate_expr = ('a*exp(-b*V)', ('a', 'b'))
    kon_rate_expr = ('k*D', ('k'))
    koff_rate_expr = ('l', ('l'))

    rate_dictionary = dict(zip(['k1', 'k3', 'k2', 'k4', 'drug_on', 'drug_off'], \
        [positive_rate_expr] * 2 + [negative_rate_expr] * 2 + [kon_rate_expr] + [koff_rate_expr]))

    mc.parameterise_rates(rate_dictionary, ['V', 'D'])
    mc.draw_graph("%s_nontrapped_open_state_blocker.html" % mc.name)

    myokitmodel = mc.get_myokit_model(drug_binding=True)
    
    if save_file:
        myokit.save(filename="%s_nontrapped_open_state_blocker.mmt" % mc.name, model=myokitmodel)


def trapped_open_inactivated_state_blocker(save_file = False):

    mc = example_models.construct_four_state_chain()
    mc.mirror_model(prefix='d_')
    drug_rates = [('s_O', 'd_s_O', 'drug_on', 'drug_off'), ('s_I', 'd_s_I', 'drug_on', 'drug_off')]

    for r in drug_rates:
        mc.add_both_transitions(*r)

    positive_rate_expr = ('a*exp(b*V)', ('a', 'b'))
    negative_rate_expr = ('a*exp(-b*V)', ('a', 'b'))
    kon_rate_expr = ('k*D', ('k'))
    koff_rate_expr = ('l', ('l'))

    rate_dictionary = dict(zip(['k1', 'k3', 'k2', 'k4', 'drug_on', 'drug_off'], \
        [positive_rate_expr] * 2 + [negative_rate_expr] * 2 + [kon_rate_expr] + [koff_rate_expr]))

    mc.parameterise_rates(rate_dictionary, ['V', 'D'])
    mc.draw_graph("%s_trapped_open_inactivated_state_blocker.html" % mc.name)

    myokitmodel = mc.get_myokit_model(drug_binding=True)
    
    if save_file:
        myokit.save(filename="%s_trapped_open_inactivated_state_blocker.mmt" % mc.name, model=myokitmodel)


def trapped_inactivated_state_blocker(save_file = False):

    mc = example_models.construct_four_state_chain()
    mc.mirror_model(prefix='d_')
    drug_rates = [('s_I', 'd_s_I', 'drug_on', 'drug_off')]

    for r in drug_rates:
        mc.add_both_transitions(*r)

    positive_rate_expr = ('a*exp(b*V)', ('a', 'b'))
    negative_rate_expr = ('a*exp(-b*V)', ('a', 'b'))
    kon_rate_expr = ('k*D', ('k'))
    koff_rate_expr = ('l', ('l'))

    rate_dictionary = dict(zip(['k1', 'k3', 'k2', 'k4', 'drug_on', 'drug_off'], \
        [positive_rate_expr] * 2 + [negative_rate_expr] * 2 + [kon_rate_expr] + [koff_rate_expr]))

    mc.parameterise_rates(rate_dictionary, ['V', 'D'])
    mc.draw_graph("%s_trapped_inactivated_state_blocker.html" % mc.name, show_parameters=True)

    myokitmodel = mc.get_myokit_model(drug_binding=True)
    print(myokitmodel.code())
    
    if save_file:
        myokit.save(filename="%s_trapped_inactivated_state_blocker.mmt" % mc.name, model=myokitmodel)


nontrapped_open_inactivated_state_blocker()
nontrapped_open_state_blocker()
trapped_open_inactivated_state_blocker()
trapped_inactivated_state_blocker()
