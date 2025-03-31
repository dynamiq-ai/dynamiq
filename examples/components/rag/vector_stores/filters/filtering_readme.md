# Filtering Logic

Technically speaking, filters are defined as nested dictionaries that can be of two types: **Comparison** or **Logic**.

## Comparison

Comparison dictionaries must contain the following keys:

- `field`
- `operator`
- `value`

The `field` value in Comparison dictionaries must be the name of one of the meta fields of a document, such as `meta.years`.

The `operator` value in Comparison dictionaries must be one of the following:

- `==`
- `!=`
- `>`
- `>=`
- `<`
- `<=`
- `in`
- `not in`

The `value` takes a single value or (in the case of `in` and `not in`) a list of values.

## Logic

Logic dictionaries must contain the following keys:

- `operator`
- `conditions`

The `conditions` key must be a list of dictionaries, either of type Comparison or Logic.

The `operator` values in Logic dictionaries must be one of the following:

- `NOT`
- `OR`
- `AND`

## Example

```
filters = {
  "operator": "AND",
  "conditions": [
    {
      "field": "years",
      "operator": "==",
      "value": "2019"
    },
    {
      "field": "companies",
      "operator": "in",
      "value": ["BMW", "Mercedes"]
    }
  ]
}
```
