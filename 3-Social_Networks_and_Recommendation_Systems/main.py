import create_graphs
import recommend_funcs
import evaluate


def main():
    # create graphs of the old and new networks
    # and select the authors with at least 10 new collaborations 2017-2018
    old_graph = create_graphs.old_networks()
    # print(old_graph.number_of_edges())
    new_graph = create_graphs.new_networks()
    # print(new_graph.number_of_edges())
    pick_list = create_graphs.pick_authors(new_graph)
    # print(pick_list)

    common_friend_accuracy = 0.0
    common_friend_rank = 0
    jaccard_index_accuracy = 0.0
    jaccard_index_rank = 0
    adamic_adar_accuracy = 0.0
    adamic_adar_rank = 0
    num_picked_author = len(pick_list)
    # recommendations using different methods
    for author in pick_list:
        # get the actually formed collaboration
        actual_network_list = evaluate.actual_network(new_graph, author)

        # recommendation by number of common friends
        common_friend_list = recommend_funcs.common_friends_number(old_graph, author)[:10]
        # print(common_friend_list)
        # calculate accuracy
        accuracy_pred = len([x for x in common_friend_list if x in actual_network_list])/len(common_friend_list)
        common_friend_accuracy = common_friend_accuracy + accuracy_pred
        # calculate rank
        author_rank = 0
        correct_num = 0
        for y in actual_network_list:
            if y in common_friend_list:
                correct_num = correct_num + 1
                author_rank = author_rank + common_friend_list.index(y)
        if correct_num != 0:
            author_rank = author_rank / correct_num
            common_friend_rank = common_friend_rank + author_rank

        # recommendation using Jaccard’s Index
        jaccard_index_list = recommend_funcs.jaccard_index(old_graph, author)[:10]
        # print(jaccard_list)
        # calculate accuracy
        accuracy_pred = len([x for x in jaccard_index_list if x in actual_network_list])/len(jaccard_index_list)
        jaccard_index_accuracy = jaccard_index_accuracy + accuracy_pred
        # calculate rank
        author_rank = 0
        correct_num = 0
        for y in actual_network_list:
            if y in jaccard_index_list:
                correct_num = correct_num + 1
                author_rank = author_rank + jaccard_index_list.index(y)
        if correct_num != 0:
            author_rank = author_rank / correct_num
            jaccard_index_rank = jaccard_index_rank + author_rank

        # recommendation using Adamic/Adar Index
        adamic_adar_list = recommend_funcs.adamic_adar_index(old_graph, author)[:10]
        # print(adamic_adar_list)
        # calculate accuracy
        accuracy_pred = len([x for x in adamic_adar_list if x in actual_network_list])/len(adamic_adar_list)
        adamic_adar_accuracy = adamic_adar_accuracy + accuracy_pred
        # calculate rank
        author_rank = 0
        correct_num = 0
        for y in actual_network_list:
            if y in adamic_adar_list:
                correct_num = correct_num + 1
                author_rank = author_rank + adamic_adar_list.index(y)
        if correct_num != 0:
            author_rank = author_rank / correct_num
            adamic_adar_rank = adamic_adar_rank + author_rank

    # output the accuracy of each recommendation function
    common_friend_average_accuracy = common_friend_accuracy / num_picked_author
    jaccard_index_average_accuracy = jaccard_index_accuracy / num_picked_author
    adamic_adar_average_accuracy = adamic_adar_accuracy / num_picked_author
    common_friend_average_rank = common_friend_rank / num_picked_author
    jaccard_index_average_rank = jaccard_index_rank / num_picked_author
    adamic_adar_average_rank = adamic_adar_rank / num_picked_author

    print("The average accuracy among users X:")
    print("Number of common friends: " + str('{:.2%}'.format(common_friend_average_accuracy)))
    print("Jaccard’s Index: " + str('{:.2%}'.format(jaccard_index_average_accuracy)))
    print("Adamic/Adar Index: " + str('{:.2%}'.format(adamic_adar_average_accuracy)))

    print("The average rank among newly formed edges:")
    print("Number of common friends: " + str('%.2f' % common_friend_average_rank))
    print("Jaccard’s Index: " + str('%.2f' % jaccard_index_average_rank))
    print("Adamic/Adar Index: " + str('%.2f' % adamic_adar_average_rank))


if __name__ == "__main__":
    main()
