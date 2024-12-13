#ifndef SALES_DATA_H
#define SALES_DATA_H

#include <iostream>

using namespace std;


class Account {
public:
	void calculate() { amount += amount * interestRate; }
	static double rate() { return interestRate; }
	static void rate(double );


private:
	std::string owner;
	double amount;
	static double interestRate;
	static double initRate();

};


class Sales_data {

	friend Sales_data add(const Sales_data&, const Sales_data&);
	friend std::istream& read(std::istream&, Sales_data&);
	friend std::ostream& print(std::ostream&, const Sales_data&);


public:
	//Sales_data() = default;
	
	explicit Sales_data(const std::string &s = "") : bookNo(s) { }
	Sales_data(const std::string& s, unsigned cnt, double price):\
		units_sold(cnt), bookNo(s), revenue(price* cnt){ }
	explicit Sales_data(std::istream& is) { read(is, *this); }

	std::string isbn() const { return this->bookNo; }
	Sales_data& combine(const Sales_data&);

private:
	double ave_price() const { return units_sold ? revenue / units_sold : 0; };
	std::string bookNo;
	unsigned units_sold = 0;
	double revenue = 0.0;

};


//Sales_data add(const Sales_data&, const Sales_data&);
//std::istream& read(std::istream&, Sales_data&);
//std::ostream& print(std::ostream&, const Sales_data&);
//

//double Sales_data::ave_price() const {
//	if (units_sold)
//		return revenue / units_sold;
//	else
//		return 0;
//}

Sales_data& Sales_data::combine(const Sales_data& rhs) {
	units_sold += rhs.units_sold;   //total.units_sold = total.units_sold  +  rhs.units_sold
	revenue += rhs.revenue;

	return *this;    //total
}


istream &read(istream& is, Sales_data& item) 
{
	double price = 0;
	is >> item.bookNo >> item.units_sold >> price;
	item.revenue = price * item.units_sold;

	return is;
}

ostream& print(ostream& os, const Sales_data& item) {
	os << item.isbn() << " " << item.units_sold << " "
		<< item.revenue << " " << item.ave_price();

	return os;
}


Sales_data add(const Sales_data& lhs, const Sales_data& rhs) {

	Sales_data sum = lhs;
	sum.combine(rhs);

	return sum;
}

//
//Sales_data::Sales_data(std::istream& is)
//{
//	read(is, *this);
//
//}


#endif

