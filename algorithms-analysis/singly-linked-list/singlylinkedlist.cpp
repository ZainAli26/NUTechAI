#include <iostream>
using namespace std;

struct Node {
    int data;
    Node* next;
};

class LinkedList {
    Node*  first;

    public:
    LinkedList() {
        first = NULL;
    }

    void add(int number) {
        Node* ptr = this->first;
        if (this->first==NULL) {
            this->first = new Node();
            this->first->data = number;
            this->first->next = NULL;
        }
        else {
            while(ptr->next!=NULL)
                ptr = ptr->next;
            Node* temp = new Node();
            temp->data = number;
            temp->next = NULL;
            ptr->next = temp;
        }
        cout << "added : " << number << endl;
    }

    void addAt(int number, int index) {
        Node* ptr = this->first;
        bool added = false;
        if(first==NULL) {
            this->first = new Node();
            if(index==0) {
                this->first->data = number;
                this->first->next = NULL;
            }
        }
        else if (index==0) {
            Node* newNode = new Node();
            newNode->data = number;
            newNode->next = ptr;
            this->first = newNode;
            added = true;
        }
        else {
            int i = 0;
            while(ptr!=NULL) {
                i++;
                if(i==index) {
                    Node* newNode = new Node();
                    newNode->data = number;
                    newNode->next = ptr->next;
                    ptr->next = newNode;
                    added = true;
                    break;
                }
                ptr=ptr->next;
            }
        }
        if(!added) {
            cout << "Not possible to insert at: " << index << endl;
        }
    }

    void remove(int number) {
        Node* ptr = this->first;
        if(ptr!= NULL && ptr->data==number) {
            this->first = ptr->next;
            delete ptr;
        }
        else {
            while(ptr!=NULL) {
            if (ptr->next!= NULL && ptr->next->data==number) {
                Node* temp = ptr->next;
                ptr->next = temp->next;
                delete temp;
            }
            ptr = ptr->next;
        }
        }
    }

    void removeAt(int index) {
        if (this->first == NULL) {
            cout << "List is empty, cannot remove at index: " << index << endl;
            return;
        }

        Node* ptr = this->first;

        // If removing the first element (head)
        if (index == 0) {
            this->first = ptr->next;  // Move the head to the next node
            delete ptr;  // Free the old head
            return;
        }

        int i = 0;
        // Traverse the list to find the node before the target index
        while (ptr != NULL && i < index - 1) {
            ptr = ptr->next;
            i++;
        }

        // If we reached the end of the list or index is out of bounds
        if (ptr == NULL || ptr->next == NULL) {
            cout << "Index out of bounds: " << index << endl;
            return;
        }

        // Node to be deleted
        Node* temp = ptr->next;
        ptr->next = temp->next;  // Bypass the node to be removed
        delete temp;  // Free memory of the node being removed
    }


    void show() {
        int i = 0;
        Node *ptr = this->first;
        while(ptr!=NULL) {
            cout << "index " << i << ": " << ptr->data << endl;
            i++;
            ptr = ptr->next;
        }
    }
};

int main(){
    cout << "Singly Linked List" << endl;
    LinkedList* linkedList = new LinkedList();
    linkedList->add(1);
    linkedList->add(2);
    linkedList->add(3);
    linkedList->add(4);
    linkedList->show();
    linkedList->remove(2);
    linkedList->remove(1);
    linkedList->show();
    linkedList->addAt(5,0);
    linkedList->addAt(6,2);
    linkedList->addAt(9,10);
    linkedList->show();
    linkedList->removeAt(3);
    linkedList->show();
}