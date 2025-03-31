## GraphRAG implementation

This repository showcases a GraphRAG implementation in a healthcare setting.

## Installation

```
direnv allow
pip install --upgrade pip && pip install pipenv
pipenv install
pipenv run python ./main.py
````

## Synthetic data

The dataset represents a healthcare company's product catalog with 30 products across 5 categories. The data includes:

Healthcare products: Medical devices and monitoring systems
Wellness products: Fitness, nutrition, and mental health offerings
Workplace products: Ergonomic office equipment and workplace wellness
Enterprise solutions: Corporate health programs and analytics platforms
Insurance plans: Various health insurance options

## Synthetic data structure

product_id:       Unique identifier for each product (P001-P030)
product_name:     Descriptive name of the product
category:         Product category (healthcare, wellness, workplace, enterprise, insurance)
price:            Product price in USD
launch_date:      Date when the product was launched
related_products: Comma-separated list of related product IDs

## References

[Blog Post](https://blog.mayflower.de/)