property(color, gray).
property(color, red).
property(color, blue).
property(color, green).
property(color, brown).
property(color, purple).
property(color, cyan).
property(color, yellow).

property(shape, cube).
property(shape, cylinder).
property(shape, sphere).
property(shape, cone).

property(size, small).
property(size, medium).
property(size, large).

property(material, rubber).
property(material, metal).

region(0).
region(1).
region(2).
region(3).

right_R(0, 0).
right_R(0, 1).
right_R(0, 2).
right_R(0, 3).

right_R(1, 1).
right_R(1, 3).

right_R(2, 0).
right_R(2, 1).
right_R(2, 2).
right_R(2, 3).

right_R(3, 1).
right_R(3, 3).

left_R(R1, R2) :- right_R(R2, R1).

front_R(0, 0).
front_R(0, 1).
front_R(0, 2).
front_R(0, 3).

front_R(1, 0).
front_R(1, 1).
front_R(1, 2).
front_R(1, 3).

front_R(2, 2).
front_R(2, 3).

front_R(3, 2).
front_R(3, 3).

behind_R(R1, R2) :- front_R(R2, R1). 


sameProperty(X1, X2, P) :- hasProperty(X1,P,V), hasProperty(X2,P,V), X1!=X2.

same_color(X,Y):- sameProperty(X, Y, color).
same_size(X,Y):- sameProperty(X, Y, size).
same_shape(X,Y):- sameProperty(X, Y, shape).
same_material(X,Y):- sameProperty(X, Y, material).
    

1{hasProperty(X, color, V):property(color, V)}1 :- object(X).
1{hasProperty(X, material, V):property(material, V)}1 :- object(X).
1{hasProperty(X, shape, V):property(shape, V)}1 :- object(X).
1{hasProperty(X, size, V):property(size, V)}1 :- object(X).
1{at(X, R): region(R)}1 :- object(X).

:- sameProperty(X1, X2, color), sameProperty(X1, X2, material), sameProperty(X1, X2, size), sameProperty(X1, X2, shape), object(X1), object(X2), X1!=X2.

exceed_region_capacity(R) :- #count{X: object(X), at(X, R)} >= 4, region(R).
:- exceed_region_capacity(_).




object(0..4).
:- object(X), at(X, 0), not hasProperty(X, size, large).
 :- object(X), at(X, 0), not hasProperty(X, color, green).
:- object(X), at(X, 0), hasProperty(X, size, large).
 :- object(X), at(X, 0), hasProperty(X, size, small).
:- object(X), at(X, 0), hasProperty(X, color, cyan).
 :- object(X), at(X, 0), hasProperty(X, color, green).
:- #count{X1, X2: sameProperty(X1, X2, size), object(X1), object(X2), at(X1, 2), at(X2, 1), hasProperty(X1, material, metal), hasProperty(X2, material, metal)}<2.
:- object(X), at(X, 1), hasProperty(X, size, large).
 :- object(X), at(X, 1), hasProperty(X, shape, cylinder).
:- object(X), at(X, 1), not hasProperty(X, material, metal), not hasProperty(X, color, red).
:- object(X), at(X, 2), hasProperty(X, shape, cube).
:- object(X), at(X, 2), not hasProperty(X, color, green), not hasProperty(X, shape, cone).
:- #count{X1, X2: sameProperty(X1, X2, material), object(X1), object(X2), at(X1, 3), at(X2, 1), hasProperty(X1, size, small), hasProperty(X2, size, small)}<2.
:- object(X), at(X, 3), not hasProperty(X, shape, cylinder), not hasProperty(X, material, metal).
:- object(X), at(X, 3), not hasProperty(X, color, brown), not hasProperty(X, material, metal).
:- #count{X1, X2: sameProperty(X1, X2, shape), object(X1), object(X2), at(X1, 2), at(X2, 1), hasProperty(X1, material, metal), hasProperty(X2, material, metal)}>=4.

#show hasProperty/3. #show at/2.