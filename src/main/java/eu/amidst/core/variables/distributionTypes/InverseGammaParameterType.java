/*
 *
 *
 *    Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.
 *    See the NOTICE file distributed with this work for additional information regarding copyright ownership.
 *    The ASF licenses this file to You under the Apache License, Version 2.0 (the "License"); you may not use
 *    this file except in compliance with the License.  You may obtain a copy of the License at
 *
 *            http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software distributed under the License is
 *    distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and limitations under the License.
 *
 *
 */

package eu.amidst.core.variables.distributionTypes;

import eu.amidst.core.distribution.ConditionalDistribution;
import eu.amidst.core.distribution.Normal;
import eu.amidst.core.exponentialfamily.EF_InverseGamma;
import eu.amidst.core.variables.DistributionType;
import eu.amidst.core.variables.Variable;

import java.util.List;

/**
 * This class extends the abstract class {@link DistributionType} and defines the Inverse Gamma parameter type.
 */
public class InverseGammaParameterType extends DistributionType {

    /**
     * Creates a new InverseGammaParameterType for the given variable.
     * @param var_ the Variable to which the Inverse Gamma distribution will be assigned.
     */
    public InverseGammaParameterType(Variable var_) {
        super(var_);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public boolean isParentCompatible(Variable parent) {
        return false;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public Normal newUnivariateDistribution() {
        throw new UnsupportedOperationException("Inverse Gamma Parameter Type does not allow standard distributions");
    }

    /**
     * Creates a new exponential family univariate distribution.
     * @return an exponential family Inverse Gamma distribution.
     */
    @Override
    public EF_InverseGamma newEFUnivariateDistribution() {
        EF_InverseGamma inverseGamma = new EF_InverseGamma(this.variable);
        inverseGamma.getNaturalParameters().set(0, -1 - 1); //alpha = 0.1
        inverseGamma.getNaturalParameters().set(1, -1);   //beta = 1
        inverseGamma.fixNumericalInstability();
        inverseGamma.updateMomentFromNaturalParameters();
        return inverseGamma;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public <E extends ConditionalDistribution> E newConditionalDistribution(List<Variable> parents) {
        throw new UnsupportedOperationException("Inverse Gamma Parameter Type does not allow conditional distributions");
    }
}
